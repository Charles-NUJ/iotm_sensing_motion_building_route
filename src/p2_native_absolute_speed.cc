// P2 fixed-prior source-route overlay on native ns-3.32 ad hoc Wi-Fi.
// GPL-2.0-or-later, consistent with the linked ns-3 dependency.
// Absolute-speed experiment with native ns-3 mobility models.
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/wifi-module.h"
#include "ns3/propagation-module.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <vector>
using namespace ns3;
using Vec=std::array<double,3>;
static Vec Add(Vec a,Vec b){for(int i=0;i<3;i++)a[i]+=b[i];return a;}
static Vec Sub(Vec a,Vec b){for(int i=0;i<3;i++)a[i]-=b[i];return a;}
static Vec Mul(Vec a,double x){for(auto &v:a)v*=x;return a;}
static double Dot(Vec a,Vec b){double v=0;for(int i=0;i<3;i++)v+=a[i]*b[i];return v;}
static double Norm(Vec a){return std::sqrt(Dot(a,a));}
static Vector NsVector(Vec a){return Vector(a[0],a[1],a[2]);}
static Vec ArrayVector(Vector a){return {a.x,a.y,a.z};}
static Vec AtSpeed(Vec desired,Vec fallback,double speed){
  if(speed<=1e-12)return {0,0,0};double magnitude=Norm(desired);
  if(magnitude<=1e-12){desired=fallback;magnitude=Norm(desired);}
  if(magnitude<=1e-12)return {speed,0,0};return Mul(desired,speed/magnitude);
}
static double T(){return Simulator::Now().GetSeconds();}
static void Require(bool condition,const char *text){if(!condition)throw std::runtime_error(text);}
static double F32(double x){return static_cast<float>(x);}
static double Half(double x){ // IEEE binary16 round-to-nearest-even, finite normal data
  float f=static_cast<float>(x);uint32_t b;std::memcpy(&b,&f,4);
  uint32_t sign=(b>>16)&0x8000, exp=(b>>23)&255, frac=b&0x7fffff;int e=int(exp)-127+15;uint32_t h;
  if(e>=31)h=sign|0x7c00;
  else if(e<=0){if(e<-10)h=sign;else{uint32_t m=frac|0x800000;int shift=14-e;uint32_t q=m>>shift,r=m&((1u<<shift)-1);q+=(r>(1u<<(shift-1))||(r==(1u<<(shift-1))&&(q&1)));h=sign|q;}}
  else{uint32_t q=frac>>13,r=frac&8191;q+=(r>4096||(r==4096&&(q&1)));h=sign+(uint32_t(e)<<10)+q;}
  uint32_t hs=h&0x8000,he=(h>>10)&31,hf=h&1023;
  double value=he==0?std::ldexp(double(hf),-24):(he==31?INFINITY:std::ldexp(1.+double(hf)/1024,int(he)-15));
  return hs?-value:value;
}
struct State{double time=0;Vec p{},v{};std::array<double,6> mean{};};
using Mat2=std::array<std::array<double,2>,2>;
static Mat2 MZero(){return {{{0,0},{0,0}}};}
static Mat2 MIdentity(){return {{{1,0},{0,1}}};}
static Mat2 MAdd(Mat2 a,const Mat2&b){for(int i=0;i<2;i++)for(int j=0;j<2;j++)a[i][j]+=b[i][j];return a;}
static Mat2 MSub(Mat2 a,const Mat2&b){for(int i=0;i<2;i++)for(int j=0;j<2;j++)a[i][j]-=b[i][j];return a;}
static Mat2 MTranspose(const Mat2&a){return {{{a[0][0],a[1][0]},{a[0][1],a[1][1]}}};}
static Mat2 MMul(const Mat2&a,const Mat2&b){Mat2 c=MZero();for(int i=0;i<2;i++)for(int k=0;k<2;k++)for(int j=0;j<2;j++)c[i][j]+=a[i][k]*b[k][j];return c;}
static Mat2 MSym(Mat2 a){double x=.5*(a[0][1]+a[1][0]);a[0][1]=a[1][0]=x;return a;}
static Mat2 ProcessCov(double dt,double accelerationStd){double q=accelerationStd*accelerationStd;return {{{q*std::pow(dt,4)/4,q*std::pow(dt,3)/2},{q*std::pow(dt,3)/2,q*dt*dt}}};}
static Mat2 PseudoInverseSymmetric(Mat2 s){
  s=MSym(s);double trace=s[0][0]+s[1][1],disc=std::sqrt(std::max(0.,(s[0][0]-s[1][1])*(s[0][0]-s[1][1])+4*s[0][1]*s[0][1]));
  double high=.5*(trace+disc),low=.5*(trace-disc);if(high<=0)return MZero();
  if(low>1e-12*high){double determinant=s[0][0]*s[1][1]-s[0][1]*s[1][0];return {{{s[1][1]/determinant,-s[0][1]/determinant},{-s[1][0]/determinant,s[0][0]/determinant}}};}
  if(disc<=1e-30)return {{{1/high,0},{0,1/high}}};
  Mat2 projector={{{(s[0][0]-low)/disc,s[0][1]/disc},{s[1][0]/disc,(s[1][1]-low)/disc}}};
  for(auto &row:projector)for(auto &value:row)value/=high;return projector;
}
struct IntermittentCvKalman{
  State state;Mat2 covariance=MZero();double filterTime=0,lastMeasurementTime=0,positionStd=0,velocityStd=0,accelerationStd=.4;uint64_t accepted=0,rejected=0;
  void Initialize(State initial,double positionNoise,double velocityNoise,double processAcceleration){state=initial;filterTime=lastMeasurementTime=initial.time;positionStd=positionNoise;velocityStd=velocityNoise;accelerationStd=processAcceleration;covariance=MZero();accepted=rejected=0;}
  void Predict(double time){Require(time>=filterTime-1e-12,"Kalman backward prediction");double dt=std::max(0.,time-filterTime);if(dt==0){filterTime=time;return;}for(int axis=0;axis<3;axis++)state.p[axis]+=dt*state.v[axis];Mat2 transition={{{1,dt},{0,1}}};covariance=MSym(MAdd(MMul(MMul(transition,covariance),MTranspose(transition)),ProcessCov(dt,accelerationStd)));filterTime=time;}
  bool UpdateDelayed(State measurement,double receptionTime){
    Require(receptionTime>=measurement.time-1e-12,"measurement received before generation");if(measurement.time<=lastMeasurementTime+1e-12){rejected++;return false;}Predict(receptionTime);double age=std::max(0.,receptionTime-measurement.time);
    Mat2 propagation={{{1,age},{0,1}}};Mat2 base={{{positionStd*positionStd,0},{0,velocityStd*velocityStd}}};Mat2 measurementCov=MSym(MAdd(MMul(MMul(propagation,base),MTranspose(propagation)),ProcessCov(age,accelerationStd)));
    Mat2 gain=MMul(covariance,PseudoInverseSymmetric(MAdd(covariance,measurementCov)));for(int axis=0;axis<3;axis++){double z0=measurement.p[axis]+age*measurement.v[axis],z1=measurement.v[axis];double e0=z0-state.p[axis],e1=z1-state.v[axis];state.p[axis]+=gain[0][0]*e0+gain[0][1]*e1;state.v[axis]+=gain[1][0]*e0+gain[1][1]*e1;}
    Mat2 remainder=MSub(MIdentity(),gain);covariance=MSym(MAdd(MMul(MMul(remainder,covariance),MTranspose(remainder)),MMul(MMul(gain,measurementCov),MTranspose(gain))));lastMeasurementTime=measurement.time;accepted++;return true;
  }
  State Snapshot(double time){Predict(time);State result=state;result.time=lastMeasurementTime;return result;}
  double Trace3d()const{return 3*(covariance[0][0]+covariance[1][1]);}
};
static std::vector<double> Features(State a,State b){
  Vec r=Sub(b.p,a.p),u=Sub(b.v,a.v);double d=Norm(r),s=Norm(u),rad=d>1e-12?Dot(r,u)/d:0;
  double aa=Dot(u,u),bb=2*Dot(r,u),cc=Dot(r,r)-40000,life=0;
  if(cc<=4e-8){if(aa<=1e-18)life=INFINITY;else{double disc=bb*bb-4*aa*cc;if(disc<0&&disc>-4e-8)disc=0;if(disc>=0)life=std::max(0.,(-bb+std::sqrt(disc))/(2*aa));}}
  return {F32(std::min(1.5,d/200)),F32(std::min(2.,s/25)),F32(std::max(-2.,std::min(2.,rad/25))),F32(std::min(2.,std::sqrt(std::max(0.,s*s-rad*rad))/25)),F32(std::isinf(life)?1:std::min(1.,life/5))};
}
static std::array<double,6> NodeRow(State a){return {F32(a.p[0]/2000),F32(a.p[1]/2000),F32(a.p[2]/120),F32(a.v[0]/25),F32(a.v[1]/25),F32(a.v[2]/25)};}
struct Fit{
  int type=0,dim=0,h1=0,h2=0;std::vector<double>w1,b1,w2,b2,w3,b3,coef;double intercept=0;
  void Read(const std::string &path){std::ifstream f(path);Require(bool(f),"model file unavailable");f>>type>>dim>>h1>>h2;
    auto values=[&](std::vector<double>&v,int n){v.resize(n);for(auto &x:v)f>>x;};
    if(!type){values(coef,dim);f>>intercept;}else{values(w1,h1*dim);values(b1,h1);values(w2,h2*h1);values(b2,h2);values(w3,h2);values(b3,1);}Require(bool(f),"model parse failed");}
  double Predict(std::vector<double>x)const{Require(int(x.size())==dim,"input dimension");
    if(!type){double q=intercept;for(int i=0;i<dim;i++)q+=coef[i]*x[i];return std::min(1.,std::max(0.,q));}
    std::vector<double>a(h1),b(h2);for(int i=0;i<h1;i++){double z=b1[i];for(int j=0;j<dim;j++)z+=w1[i*dim+j]*x[j];a[i]=std::max(0.,z);}
    for(int i=0;i<h2;i++){double z=b2[i];for(int j=0;j<h1;j++)z+=w2[i*h1+j]*a[j];b[i]=std::max(0.,z);}
    double z=b3[0];for(int j=0;j<h2;j++)z+=w3[j]*b[j];return 1/(1+std::exp(-std::max(-60.,std::min(60.,z))));}
};
enum Kind{HELLO=1,RREQ=2,RREP=3,DATA=4,RERR=5,OTHER=6,ACK=7};
struct TraceTag:public Tag{
  uint8_t kind=OTHER;static TypeId GetTypeId(){static TypeId tid=TypeId("P2NativeTraceTag").SetParent<Tag>().AddConstructor<TraceTag>();return tid;}
  TypeId GetInstanceTypeId()const override{return GetTypeId();}uint32_t GetSerializedSize()const override{return 1;}
  void Serialize(TagBuffer b)const override{b.WriteU8(kind);}void Deserialize(TagBuffer b)override{kind=b.ReadU8();}
  void Print(std::ostream&o)const override{o<<int(kind);}
};
struct Bytes{
  std::vector<uint8_t>b;size_t at=0;
  void Put(uint64_t x,int n){for(int i=0;i<n;i++)b.push_back((x>>(8*i))&255);}
  uint64_t Get(int n){Require(at+n<=b.size(),"short packet");uint64_t x=0;for(int i=0;i<n;i++)x|=uint64_t(b[at++])<<(8*i);return x;}
  void D(double x){uint64_t u;std::memcpy(&u,&x,8);Put(u,8);}double D(){uint64_t u=Get(8);double x;std::memcpy(&x,&u,8);return x;}
  void F(double x){float f=x;uint32_t u;std::memcpy(&u,&f,4);Put(u,4);}double F(){uint32_t u=Get(4);float f;std::memcpy(&f,&u,4);return f;}
  void Pad(size_t n){Require(b.size()<=n,"header budget overflow");b.resize(n);}
};
struct Message{Kind kind=OTHER;uint32_t src=0,dst=0,id=0,route=0,index=0;double generated=0,cost=0;std::vector<uint32_t>path;State state;};
static Bytes Encode(Message m){Bytes out;out.Put(m.kind,1);
  if(m.kind==HELLO){out.Put(m.src,2);out.Put(m.id,4);out.D(m.state.time);for(double x:m.state.p)out.F(x);for(double x:m.state.v)out.F(x);
    // Six binary16 summaries occupy 12 bytes. Encode their exact binary16 values
    // as an explicit 16-bit sign/exponent/mantissa, not extra simulator metadata.
    for(double x:m.state.mean){double q=Half(x);uint16_t bits=0;for(int sign=0;sign<2;sign++){if((q<0)==bool(sign)){double a=std::abs(q);if(a==0)bits=uint16_t(sign<<15);else if(a<std::ldexp(1.,-14))bits=(sign<<15)|uint16_t(std::llround(std::ldexp(a,24)));else{int e;double f=std::frexp(a,&e);bits=(sign<<15)|((e+14)<<10)|uint16_t(std::llround((f*2-1)*1024));}break;}}out.Put(bits,2);}out.Pad(68);return out;}
  out.Put(m.src,2);out.Put(m.dst,2);out.Put(m.id,4);
  if(m.kind==RREQ){out.D(m.cost);out.Put(m.path.size(),1);out.D(m.generated);out.Pad(28);}
  if(m.kind==RREP){out.Put(m.path.size(),1);out.Put(m.index,2);out.D(m.cost);out.Pad(20);}
  if(m.kind==DATA){out.D(m.generated);out.Put(m.index,2);out.Put(m.path.size(),1);out.Put(m.route,4);out.Pad(24);}
  if(m.kind==RERR){out.Put(m.route,4);out.Pad(16);return out;}
  for(auto node:m.path)out.Put(node,4);
  if(m.kind==DATA){size_t target=out.b.size()+512;for(auto x:m.state.p)out.D(x);for(auto x:m.state.v)out.D(x);out.Pad(target);}return out;
}
static Message Decode(Bytes in){Message m;m.kind=Kind(in.Get(1));
  if(m.kind==HELLO){m.src=in.Get(2);m.id=in.Get(4);m.state.time=in.D();for(auto &x:m.state.p)x=in.F();for(auto &x:m.state.v)x=in.F();
    for(auto &x:m.state.mean){uint32_t h=in.Get(2),e=(h>>10)&31,v=h&1023;double q=e==0?std::ldexp(double(v),-24):std::ldexp(1.+double(v)/1024,int(e)-15);x=(h&0x8000)?-q:q;}Require(in.b.size()==68,"HELLO length");return m;}
  m.src=in.Get(2);m.dst=in.Get(2);m.id=in.Get(4);uint32_t count=0;
  if(m.kind==RREQ){m.cost=in.D();count=in.Get(1);m.generated=in.D();in.at=28;}
  if(m.kind==RREP){count=in.Get(1);m.index=in.Get(2);m.cost=in.D();in.at=20;}
  if(m.kind==DATA){m.generated=in.D();m.index=in.Get(2);count=in.Get(1);m.route=in.Get(4);in.at=24;}
  if(m.kind==RERR){m.route=in.Get(4);Require(in.b.size()==16,"RERR length");return m;}
  Require(count>=1&&count<=64,"path length");for(uint32_t j=0;j<count;j++)m.path.push_back(in.Get(4));
  if(m.kind==DATA){
    // The DATA creation time is also the source-state measurement time.  The
    // state timestamp is not carried twice in the packet, so restore it after
    // decoding before the estimator performs its stale-measurement check.
    m.state.time=m.generated;
    for(auto &x:m.state.p)x=in.D();for(auto &x:m.state.v)x=in.D();
  }return m;
}
struct Route{uint32_t id=0;double expires=0;std::vector<uint32_t>path;bool broken=false;};
struct Label{double cost=INFINITY;std::vector<uint32_t>path;bool pending=false,replied=false;uint64_t replyGeneration=0;};
static bool BetterLabel(double cost,const std::vector<uint32_t>&path,const Label&label){
  if(cost<label.cost-1e-12)return true;
  if(std::abs(cost-label.cost)>1e-12)return false;
  if(path.size()!=label.path.size())return path.size()<label.path.size();
  return std::lexicographical_compare(path.begin(),path.end(),label.path.begin(),label.path.end());
}
class Scenario{
public:
  uint32_t n=24,seed=6301;std::string method="hop",mission="waypoint",adaptationBand="",input="inputs",output="raw/run",estimator="kf";bool mute=false,outage=false,summaryOnly=false,mobilityOnly=false;
  double absoluteSpeed=0.,formationRelativeSpeedTarget=-1.;
  double formationExcitationPeriod=40.,formationExcitationSharpness=1.;
  double positionMeasurementStd=.25,velocityMeasurementStd=.03,processAccelerationStd=2.;
  std::vector<State> initial;
  NodeContainer nodes;NetDeviceContainer devices;Ipv4InterfaceContainer ip;std::vector<Ptr<Socket>>sockets;
  std::vector<std::map<uint32_t,State>>neighbors;std::vector<std::map<uint32_t,Label>>labels;
  std::vector<std::map<uint32_t,Route>>localRoutes;std::map<uint32_t,Route>sourceRoutes;
  std::map<uint32_t,std::vector<Message>>pending;std::map<uint32_t,uint32_t>discovering;
  std::map<uint32_t,double>sourceStarted;std::map<uint32_t,Label>sourceReplyLabels;std::set<uint32_t>sourceCommitScheduled;std::vector<State>est;std::vector<IntermittentCvKalman>filters;std::vector<uint32_t>destinations;
  Ptr<NormalRandomVariable> observationNoise;
  std::map<Mac48Address,uint32_t>macNode;Fit fit;Ptr<UniformRandomVariable> jitter;
  std::ofstream tx,plant,mobilityTrace,deliveries,events,receives,macok;std::vector<double>txBegin;std::vector<uint32_t>txBytes,txKind,txSeq;std::vector<Mac48Address>txPeer;
  uint32_t discoverySeq=0,packetSeq=0,helloSeq=0,offered=0,delivered=0,failures=0,rerrRx=0;
  uint64_t phyBytes=0,phyControlBytes=0,upperControlBytes=0,ackBytes=0,phyFrames=0,ackFrames=0,observations=0;
  uint64_t macOkHeaders=0,unicastPhyAttempts=0,groupPhyAttempts=0,rxDropEvents=0;
  uint64_t backoffDraws=0,backoffSlots=0,cwChanges=0;uint32_t maxCwObserved=0;double summedTxAirtime=0;
  std::map<int,double>phyStateSeconds;std::map<int,uint64_t>rxDropReasons;
  std::array<uint64_t,8>kindBytes{},kindFrames{};double squaredError=0,squaredPositionEstimateError=0,squaredVelocityEstimateError=0,summedCovarianceTrace=0,observationNoiseChecksum=0;uint64_t errors=0,estimateErrors=0,observationNoiseScalars=0;std::vector<double>delays,ages,absoluteSpeedSamples,relativeSpeedSamples;uint32_t minHops=100000;
  double maximumAbsoluteSpeedError=0.;
  State Own(uint32_t i){auto model=nodes.Get(i)->GetObject<MobilityModel>();Require(bool(model),"mobility model unavailable");auto p=model->GetPosition(),v=model->GetVelocity();State s;s.time=T();s.p={p.x,p.y,p.z};s.v={v.x,v.y,v.z};return s;}
  std::array<double,6> Mean(uint32_t i){std::array<double,6>v{};int count=0;for(auto &entry:neighbors[i])if(T()-entry.second.time<=2.5){auto x=NodeRow(entry.second);for(int j=0;j<6;j++)v[j]+=x[j];count++;}for(auto &x:v)x=count?F32(x/count):0;return v;}
  double Cost(uint32_t sender,uint32_t receiver){auto found=neighbors[receiver].find(sender);if(found==neighbors[receiver].end())return INFINITY;State a=found->second,b=Own(receiver);
    Require(a.time<=T()+1e-9,"future HELLO");if(T()-a.time>2.5||Norm(Sub(a.p,b.p))>200)return INFINITY;observations++;
    if(method=="hop")return 1.;auto x=Features(a,b);double q=x[4];
    if(method=="frozen"||method=="pooled"||method=="family_adapted")q=fit.Predict(x);
    if(Norm(Sub(a.v,b.v))<=1e-12)q=1;Require(std::isfinite(q)&&q>=0&&q<=1,"invalid score");return .03-.7*std::log(std::max(.001,q));}
  void Send(uint32_t node,int next,Message message){auto bytes=Encode(message);Ptr<Packet>p=Create<Packet>(bytes.b.data(),bytes.b.size());TraceTag tag;tag.kind=message.kind;p->AddPacketTag(tag);
    if(message.kind!=DATA)upperControlBytes+=bytes.b.size();Ipv4Address dst=next<0?Ipv4Address("255.255.255.255"):ip.GetAddress(next);
    sockets[node]->SendTo(p,0,InetSocketAddress(dst,9000));}
  void Hello(uint32_t i){Message m;m.kind=HELLO;m.src=i;m.id=++helloSeq;m.state=Own(i);m.state.mean=Mean(i);Send(i,-1,m);if(T()+1<40)Simulator::Schedule(Seconds(1),&Scenario::Hello,this,i);}
  void Flush(uint32_t node,Message message){auto &label=labels[node][message.id];label.pending=false;if(T()>=message.generated+.30)return;message.cost=label.cost;message.path=label.path;Send(node,-1,message);}
  void Request(uint32_t node,Message m){if(T()>=m.generated+.30||std::find(m.path.begin(),m.path.end(),node)!=m.path.end())return;
    double edge=Cost(m.path.back(),node);if(!std::isfinite(edge))return;m.cost+=edge;m.path.push_back(node);auto &label=labels[node][m.id];
    if(!BetterLabel(m.cost,m.path,label))return;label.cost=m.cost;label.path=m.path;
    if(node==m.dst){label.replied=false;uint64_t generation=++label.replyGeneration;Simulator::Schedule(MilliSeconds(10),&Scenario::Reply,this,node,m,generation);}
    else if(!label.pending){label.pending=true;Simulator::Schedule(MilliSeconds(1+jitter->GetValue(0,2)),&Scenario::Flush,this,node,m);}}
  void Reply(uint32_t node,Message m,uint64_t generation){if(T()>=m.generated+.32)return;auto &l=labels[node][m.id];if(generation!=l.replyGeneration||l.replied)return;l.replied=true;m.kind=RREP;m.cost=l.cost;m.path=l.path;m.index=m.path.size()-2;
    localRoutes[node][m.id]={m.id,T()+6,m.path,false};Send(node,m.path[m.index],m);}
  void CommitReply(uint32_t destination,uint32_t id){sourceCommitScheduled.erase(id);if(discovering.count(destination)==0||discovering[destination]!=id)return;auto it=sourceReplyLabels.find(id);if(it==sourceReplyLabels.end()||!std::isfinite(it->second.cost)||it->second.path.empty())return;
    auto path=it->second.path;discovering.erase(destination);sourceReplyLabels.erase(it);sourceRoutes[destination]={id,T()+5,path,false};localRoutes[0][id]={id,T()+6,path,false};events<<T()<<",installed,"<<id<<","<<destination<<","<<path.size()-1<<"\n";
    auto queue=pending[destination];pending.erase(destination);for(auto data:queue)Forward(0,data,sourceRoutes[destination]);}
  void ReceiveReply(uint32_t node,Message m){Require(m.index<m.path.size()&&m.path[m.index]==node,"RREP wrong path");localRoutes[node][m.id]={m.id,T()+6,m.path,false};
    if(m.index==0){if(discovering.count(m.dst)==0||discovering[m.dst]!=m.id||T()>=sourceStarted[m.id]+.35)return;auto &candidate=sourceReplyLabels[m.id];if(!BetterLabel(m.cost,m.path,candidate))return;candidate.cost=m.cost;candidate.path=m.path;
      if(!sourceCommitScheduled.count(m.id)){sourceCommitScheduled.insert(m.id);double delay=std::max(0.,sourceStarted[m.id]+.34-T());Simulator::Schedule(Seconds(delay),&Scenario::CommitReply,this,m.dst,m.id);}}
    else{m.index--;Send(node,m.path[m.index],m);}}
  void Timeout(uint32_t destination,uint32_t id){if(discovering.count(destination)&&discovering[destination]==id){discovering.erase(destination);sourceReplyLabels.erase(id);sourceCommitScheduled.erase(id);pending.erase(destination);events<<T()<<",timeout,"<<id<<","<<destination<<",0\n";}}
  void Discover(uint32_t destination){Message m;m.kind=RREQ;m.src=0;m.dst=destination;m.id=++discoverySeq;m.generated=T();m.path={0};labels[0][m.id].cost=0;labels[0][m.id].path={0};
    discovering[destination]=m.id;sourceStarted[m.id]=T();Send(0,-1,m);Simulator::Schedule(MilliSeconds(360),&Scenario::Timeout,this,destination,m.id);}
  void Offer(uint32_t d){Message m;m.kind=DATA;m.src=0;m.dst=d;m.id=packetSeq++;m.generated=T();m.state=NoisySourceObservation();offered++;
    if(sourceRoutes.count(d)&&sourceRoutes[d].expires>=T()&&!sourceRoutes[d].broken)Forward(0,m,sourceRoutes[d]);else{pending[d].push_back(m);if(!discovering.count(d))Discover(d);}
    if(T()+.5<39)Simulator::Schedule(Seconds(.5),&Scenario::Offer,this,d);}
  void Forward(uint32_t node,Message m,Route route){if(T()-m.generated>1)return;m.path=route.path;m.route=route.id;auto here=std::find(m.path.begin(),m.path.end(),node);Require(here!=m.path.end(),"DATA path membership");
    uint32_t index=here-m.path.begin();localRoutes[node][route.id]={route.id,T()+6,route.path,route.broken};
    if(route.broken){Error(node,route);return;}if(index+1<m.path.size()){m.index=index+1;Send(node,m.path[m.index],m);}}
  void Error(uint32_t node,Route route){auto &r=localRoutes[node][route.id];if(r.broken)return;r.broken=true;auto it=std::find(route.path.begin(),route.path.end(),node);if(it==route.path.end())return;
    if(it==route.path.begin()){auto d=route.path.back();if(sourceRoutes.count(d)&&sourceRoutes[d].id==route.id)sourceRoutes.erase(d);return;}
    Message e;e.kind=RERR;e.src=0;e.dst=route.path.back();e.route=route.id;e.id=route.id;Send(node,*(it-1),e);}
  void ReceiveError(uint32_t node,Message m){rerrRx++;auto it=localRoutes[node].find(m.route);if(it!=localRoutes[node].end())Error(node,it->second);}
  std::set<uint32_t>seenData;
  void ReceiveData(uint32_t node,Message m){Require(m.index<m.path.size()&&m.path[m.index]==node,"DATA wrong path");if(T()-m.generated>1)return;
    if(node==m.dst){if(!seenData.insert(m.id).second)return;delivered++;minHops=std::min(minHops,m.index);delays.push_back(T()-m.generated);
      deliveries<<T()<<","<<m.id<<","<<m.dst<<","<<m.generated<<","<<m.index<<"\n";
      if(!mute&&m.generated>est[node].time){if(estimator=="kf")filters[node].UpdateDelayed(m.state,T());else{est[node]=m.state;est[node].time=m.generated;}}}
    else{Route r{m.route,T()+6,m.path,false};if(localRoutes[node].count(m.route))r.broken=localRoutes[node][m.route].broken;Forward(node,m,r);}}
  void Receive(uint32_t node,Ptr<Socket>socket){Address from;while(auto p=socket->RecvFrom(from)){Bytes bytes;bytes.b.resize(p->GetSize());p->CopyData(bytes.b.data(),bytes.b.size());Message m=Decode(bytes);receives<<T()<<","<<node<<","<<m.kind<<","<<m.id<<","<<p->GetUid()<<"\n";
      if(m.kind==HELLO){if(m.src!=node){Require(m.state.time<=T()+1e-9,"future HELLO timestamp");neighbors[node][m.src]=m.state;}}
      else if(m.kind==RREQ)Request(node,m);else if(m.kind==RREP)ReceiveReply(node,m);else if(m.kind==DATA)ReceiveData(node,m);else if(m.kind==RERR)ReceiveError(node,m);else Require(false,"unknown type");}}
  static void Rx(Scenario*s,uint32_t n,Ptr<Socket>socket){s->Receive(n,socket);}
  void MacFailure(uint32_t node,const WifiMacHeader&header){if(header.GetAddr1().IsGroup())return;auto found=macNode.find(header.GetAddr1());if(found==macNode.end())return;uint32_t peer=found->second;failures++;
    std::vector<Route>bad;for(auto &item:localRoutes[node]){auto route=item.second;auto here=std::find(route.path.begin(),route.path.end(),node);if(route.expires>=T()&&!route.broken&&here!=route.path.end()&&here+1!=route.path.end()&&*(here+1)==peer)bad.push_back(route);}
    for(auto route:bad)Error(node,route);}
  static void Fail(Scenario*s,uint32_t n,const WifiMacHeader&h){s->MacFailure(n,h);}
  static void Ok(Scenario*s,uint32_t node,const WifiMacHeader&h){
    // TxOkHeader is also emitted for group-addressed frames.  Keep the
    // success counter on the same unicast population as PhyTxBegin and
    // TxErrHeader so that the retry estimate has a valid denominator.
    if(h.GetAddr1().IsGroup())return;
    s->macOkHeaders++;
    s->macok<<T()<<","<<node<<","<<h.GetAddr1()<<","<<h.GetSequenceNumber()<<"\n";
  }
  static void TxBegin(Scenario*s,uint32_t node,Ptr<const Packet>p,double powerW){TraceTag tag;WifiMacHeader header;uint32_t kind=OTHER;p->PeekHeader(header);if(header.IsAck())kind=ACK;else if(p->PeekPacketTag(tag))kind=tag.kind;
    if(!header.IsAck()){if(header.GetAddr1().IsGroup())s->groupPhyAttempts++;else s->unicastPhyAttempts++;}
    s->txBegin[node]=T();s->txBytes[node]=p->GetSize();s->txKind[node]=kind;s->txSeq[node]=header.GetSequenceNumber();s->txPeer[node]=header.GetAddr1();s->phyBytes+=p->GetSize();s->phyFrames++;s->kindBytes[kind]+=p->GetSize();s->kindFrames[kind]++;
    if(kind!=DATA)s->phyControlBytes+=p->GetSize();if(kind==ACK){s->ackBytes+=p->GetSize();s->ackFrames++;}}
  static void TxEnd(Scenario*s,uint32_t node,Ptr<const Packet>p){s->summedTxAirtime+=T()-s->txBegin[node];s->tx<<s->txBegin[node]<<","<<T()<<","<<node<<","<<s->txKind[node]<<","<<s->txBytes[node]<<","<<p->GetUid()<<","<<s->txSeq[node]<<","<<s->txPeer[node]<<"\n";}
  static void StateLog(Scenario*s,uint32_t node,Time start,Time duration,WifiPhyState state){s->phyStateSeconds[int(state)]+=duration.GetSeconds();}
  static void BackoffLog(Scenario*s,uint32_t node,uint32_t slots){s->backoffDraws++;s->backoffSlots+=slots;}
  static void CwLog(Scenario*s,uint32_t node,uint32_t oldValue,uint32_t newValue){s->cwChanges++;s->maxCwObserved=std::max(s->maxCwObserved,newValue);}
  static void DropLog(Scenario*s,uint32_t node,Ptr<const Packet>packet,WifiPhyRxfailureReason reason){s->rxDropEvents++;s->rxDropReasons[int(reason)]++;}
  void Outage(){if(sourceRoutes.empty()){events<<T()<<",outage_no_route,0,0,0\n";return;}auto path=sourceRoutes.begin()->second.path;Require(path.size()>=4,"outage needs multihop");uint32_t node=path[path.size()/2];auto phy=DynamicCast<WifiNetDevice>(devices.Get(node))->GetPhy();phy->SetOffMode();events<<T()<<",radio_off,"<<node<<",0,0\n";Simulator::Schedule(Seconds(5),&Scenario::Restore,this,node);}
  void Restore(uint32_t node){DynamicCast<WifiNetDevice>(devices.Get(node))->GetPhy()->ResumeFromOff();events<<T()<<",radio_on,"<<node<<",0,0\n";}
  NodeContainer Group(const std::vector<uint32_t>&indices){NodeContainer group;for(auto i:indices)group.Add(nodes.Get(i));return group;}
  Ptr<ListPositionAllocator> Positions(const std::vector<uint32_t>&indices){auto allocator=CreateObject<ListPositionAllocator>();for(auto i:indices)allocator->Add(NsVector(initial[i].p));return allocator;}
  std::vector<uint32_t> Sequence(uint32_t first,uint32_t last){std::vector<uint32_t> result;for(uint32_t i=first;i<last;i++)result.push_back(i);return result;}
  std::string ConstantRv(double value){std::ostringstream out;out<<std::setprecision(17)<<value;return "ns3::ConstantRandomVariable[Constant="+out.str()+"]";}
  void BuildInitial(){uint32_t rows=n/6;Require(rows>=3,"at least three rows required");initial.resize(n);for(uint32_t i=0;i<n;i++){double x=60.+120.*(i%6),y=60.+300.*(i/6)/(rows-1);initial[i].time=0;initial[i].p={x,y,60.};initial[i].v={0,0,0};}destinations={n-1,n-2};}
  void InstallConstant(const std::vector<uint32_t>&indices,const std::string&type){if(indices.empty())return;MobilityHelper helper;helper.SetPositionAllocator(Positions(indices));helper.SetMobilityModel(type);auto group=Group(indices);helper.Install(group);}
  void InstallMobility(){
    auto all=Sequence(0,n),autonomous=Sequence(0,n-2),controlled=Sequence(n-2,n);
    if(absoluteSpeed<=1e-12){InstallConstant(all,"ns3::ConstantPositionMobilityModel");return;}
    if(mission=="formation"){
      InstallConstant(all,"ns3::ConstantVelocityMobilityModel");double heading=std::fmod(.001*seed,6.283185307179586);Vec velocity={absoluteSpeed*std::cos(heading),absoluteSpeed*std::sin(heading),0};
      for(uint32_t i=0;i<n;i++)nodes.Get(i)->GetObject<ConstantVelocityMobilityModel>()->SetVelocity(NsVector(velocity));return;
    }
    if(mission=="waypoint"){
      auto targets=CreateObject<RandomBoxPositionAllocator>();auto x=CreateObject<UniformRandomVariable>(),y=CreateObject<UniformRandomVariable>();auto z=CreateObject<ConstantRandomVariable>();x->SetAttribute("Min",DoubleValue(0));x->SetAttribute("Max",DoubleValue(720));y->SetAttribute("Min",DoubleValue(0));y->SetAttribute("Max",DoubleValue(420));z->SetAttribute("Constant",DoubleValue(60));targets->SetX(x);targets->SetY(y);targets->SetZ(z);
      x->SetStream(3001);y->SetStream(3002);MobilityHelper helper;helper.SetPositionAllocator(Positions(autonomous));helper.SetMobilityModel("ns3::RandomWaypointMobilityModel","Speed",StringValue(ConstantRv(absoluteSpeed)),"Pause",StringValue(ConstantRv(0.)),"PositionAllocator",PointerValue(targets));auto group=Group(autonomous);helper.Install(group);helper.AssignStreams(group,1000);
    }else{
      MobilityHelper helper;helper.SetPositionAllocator(Positions(autonomous));helper.SetMobilityModel("ns3::GaussMarkovMobilityModel","Bounds",BoxValue(Box(0,720,0,420,59,61)),"TimeStep",TimeValue(Seconds(.5)),"Alpha",DoubleValue(.85),"MeanVelocity",StringValue(ConstantRv(absoluteSpeed)),"MeanDirection",StringValue("ns3::UniformRandomVariable[Min=0|Max=6.283185307179586]"),"MeanPitch",StringValue(ConstantRv(0.)),"NormalVelocity",StringValue("ns3::NormalRandomVariable[Mean=0|Variance=0|Bound=0]"),"NormalDirection",StringValue("ns3::NormalRandomVariable[Mean=0|Variance=0.04|Bound=0.4]"),"NormalPitch",StringValue("ns3::NormalRandomVariable[Mean=0|Variance=0|Bound=0]"));auto group=Group(autonomous);helper.Install(group);helper.AssignStreams(group,2000);
    }
    InstallConstant(controlled,"ns3::ConstantVelocityMobilityModel");
  }
  void InitializeState(){est.resize(n);filters.resize(n);State source=Own(0);for(auto d:destinations){est[d]=source;est[d].time=0;filters[d].Initialize(est[d],positionMeasurementStd,velocityMeasurementStd,processAccelerationStd);}}
  void LoadFit(){if(method=="frozen")fit.Read(input+"/frozen.dat");else if(method=="pooled")fit.Read(input+"/pooled.dat");else if(method=="family_adapted")fit.Read(input+"/family_"+mission+"_"+adaptationBand+".dat");}
  void ControlMotion(std::map<uint32_t,State>&controllerEstimate){double t=T();if(absoluteSpeed<=1e-12)return;State source=Own(0);
    if(mission=="formation"){
      double heading=std::fmod(.001*seed+.04*t,6.283185307179586);Vec leader={absoluteSpeed*std::cos(heading),absoluteSpeed*std::sin(heading),0};nodes.Get(0)->GetObject<ConstantVelocityMobilityModel>()->SetVelocity(NsVector(leader));source=Own(0);
      double ratio=formationRelativeSpeedTarget/(2*absoluteSpeed),theta=2*std::asin(std::max(0.,std::min(1.,ratio)));
      double smooth=std::tanh(formationExcitationSharpness*std::sin(2*3.14159265358979323846*t/formationExcitationPeriod))/std::tanh(formationExcitationSharpness);
      for(uint32_t i=1;i<n-2;i++){double sign=(i%2)?1.:-1.,relayHeading=heading+sign*.5*theta*smooth;Vec desired={absoluteSpeed*std::cos(relayHeading),absoluteSpeed*std::sin(relayHeading),0};auto model=nodes.Get(i)->GetObject<ConstantVelocityMobilityModel>();model->SetVelocity(NsVector(AtSpeed(desired,leader,absoluteSpeed)));}
    }
    for(auto d:destinations){State old=Own(d),e=estimator=="kf"?filters[d].Snapshot(t):est[d];controllerEstimate[d]=e;Vec target=estimator=="kf"?e.p:Add(e.p,Mul(e.v,std::max(0.,t-e.time))),offset=Sub(initial[d].p,initial[0].p),desired=Add(e.v,Mul(Sub(Add(target,offset),old.p),.6));auto model=nodes.Get(d)->GetObject<ConstantVelocityMobilityModel>();Require(bool(model),"controlled node requires ConstantVelocityMobilityModel");model->SetVelocity(NsVector(AtSpeed(desired,old.v,absoluteSpeed)));}
  }
  State NoisySourceObservation(){State measurement=Own(0);for(auto &x:measurement.p){double noise=observationNoise->GetValue();observationNoiseScalars++;observationNoiseChecksum+=observationNoiseScalars*noise;x+=positionMeasurementStd*noise;}for(auto &x:measurement.v){double noise=observationNoise->GetValue();observationNoiseScalars++;observationNoiseChecksum+=observationNoiseScalars*noise;x+=velocityMeasurementStd*noise;}return measurement;}
  void MobilityOnlyObservation(){Require(mobilityOnly,"mobility-only observation outside collection mode");for(auto d:destinations){State measurement=NoisySourceObservation();if(estimator=="kf")filters[d].UpdateDelayed(measurement,T());else est[d]=measurement;}if(T()+.5<40)Simulator::Schedule(Seconds(.5),&Scenario::MobilityOnlyObservation,this);}
  void SampleMotion(){double t=T();std::vector<State>state(n);for(uint32_t i=0;i<n;i++){state[i]=Own(i);if(mobilityOnly)mobilityTrace<<t<<","<<i<<","<<state[i].p[0]<<","<<state[i].p[1]<<","<<state[i].p[2]<<","<<state[i].v[0]<<","<<state[i].v[1]<<","<<state[i].v[2]<<"\n";if(t>=.1){double value=Norm(state[i].v);absoluteSpeedSamples.push_back(value);maximumAbsoluteSpeedError=std::max(maximumAbsoluteSpeedError,std::abs(value-absoluteSpeed));}}
    if(t>=.1)for(uint32_t i=0;i<n;i++)for(uint32_t j=i+1;j<n;j++)if(Norm(Sub(state[i].p,state[j].p))<=200.)relativeSpeedSamples.push_back(Norm(Sub(state[i].v,state[j].v)));}
  void Plant(){double t=T();std::map<uint32_t,State>controllerEstimate;ControlMotion(controllerEstimate);SampleMotion();State source=Own(0);
    if(!mobilityOnly)for(auto d:destinations){State state=Own(d),e=controllerEstimate.count(d)?controllerEstimate[d]:(estimator=="kf"?filters[d].Snapshot(t):est[d]);Vec error=Sub(Sub(state.p,source.p),Sub(initial[d].p,initial[0].p));double value=Norm(error),positionError=Norm(Sub(e.p,source.p)),velocityError=Norm(Sub(e.v,source.v));plant<<t<<","<<d<<","<<value<<","<<e.time<<","<<positionError<<","<<velocityError;for(auto x:state.p)plant<<","<<x;for(auto x:state.v)plant<<","<<x;plant<<"\n";if(t>=3){squaredError+=value*value;errors++;ages.push_back(std::max(0.,t-e.time));squaredPositionEstimateError+=positionError*positionError;squaredVelocityEstimateError+=velocityError*velocityError;estimateErrors++;if(estimator=="kf")summedCovarianceTrace+=filters[d].Trace3d();}}
    if(t+.1<40)Simulator::Schedule(Seconds(.1),&Scenario::Plant,this);}
  void Setup(){Require(n>=18&&n%6==0,"supported node count");Require(absoluteSpeed>=0&&absoluteSpeed<=5,"absolute speed must be in [0,5] m/s");if(mission=="formation"){Require(formationRelativeSpeedTarget>=0,"formation requires a preregistered relative-speed target");Require(formationRelativeSpeedTarget<=2*absoluteSpeed+1e-12,"formation relative-speed target violates r <= 2s");Require(formationExcitationPeriod>0&&formationExcitationSharpness>0,"formation excitation parameters must be positive");}else Require(formationRelativeSpeedTarget<0,"relative-speed excitation is formation-only");BuildInitial();nodes.Create(n);InstallMobility();InitializeState();observationNoise=CreateObject<NormalRandomVariable>();observationNoise->SetAttribute("Mean",DoubleValue(0.));observationNoise->SetAttribute("Variance",DoubleValue(1.));observationNoise->SetStream(9001);if(!mobilityOnly)LoadFit();
    if(mobilityOnly){mobilityTrace.open(output+".mobility.csv");Require(bool(mobilityTrace),"mobility trace unavailable");mobilityTrace<<std::setprecision(17)<<"time_s,node,x,y,z,vx,vy,vz\n";Simulator::Schedule(MicroSeconds(1),&Scenario::Plant,this);Simulator::Schedule(MilliSeconds(250),&Scenario::MobilityOnlyObservation,this);return;}
    YansWifiChannelHelper channel;channel.SetPropagationDelay("ns3::ConstantSpeedPropagationDelayModel");channel.AddPropagationLoss("ns3::LogDistancePropagationLossModel","Exponent",DoubleValue(2.2),"ReferenceLoss",DoubleValue(46.6777));
    YansWifiPhyHelper phy=YansWifiPhyHelper::Default();phy.SetChannel(channel.Create());phy.Set("TxPowerStart",DoubleValue(16));phy.Set("TxPowerEnd",DoubleValue(16));phy.Set("RxNoiseFigure",DoubleValue(5));phy.Set("RxSensitivity",DoubleValue(-96));phy.Set("CcaEdThreshold",DoubleValue(-82));phy.SetErrorRateModel("ns3::NistErrorRateModel");
    WifiHelper wifi;wifi.SetStandard(WIFI_STANDARD_80211a);wifi.SetRemoteStationManager("ns3::ConstantRateWifiManager","DataMode",StringValue("OfdmRate6Mbps"),"ControlMode",StringValue("OfdmRate6Mbps"),"NonUnicastMode",StringValue("OfdmRate6Mbps"));
    WifiMacHelper mac;mac.SetType("ns3::AdhocWifiMac");devices=wifi.Install(phy,mac,nodes);
    Config::Set("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/RemoteStationManager/MaxSsrc",UintegerValue(3));Config::Set("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/RemoteStationManager/MaxSlrc",UintegerValue(3));
    InternetStackHelper internet;internet.Install(nodes);Ipv4AddressHelper address;address.SetBase("10.71.0.0","255.255.255.0");ip=address.Assign(devices);
    neighbors.resize(n);labels.resize(n);localRoutes.resize(n);txBegin.resize(n);txBytes.resize(n);txKind.resize(n);txSeq.resize(n);txPeer.resize(n);jitter=CreateObject<UniformRandomVariable>();jitter->SetStream(8001);
    sockets.resize(n);for(uint32_t i=0;i<n;i++){auto dev=DynamicCast<WifiNetDevice>(devices.Get(i));macNode[Mac48Address::ConvertFrom(dev->GetAddress())]=i;
      dev->GetPhy()->TraceConnectWithoutContext("PhyTxBegin",MakeBoundCallback(&Scenario::TxBegin,this,i));dev->GetPhy()->TraceConnectWithoutContext("PhyTxEnd",MakeBoundCallback(&Scenario::TxEnd,this,i));dev->GetMac()->TraceConnectWithoutContext("TxErrHeader",MakeBoundCallback(&Scenario::Fail,this,i));dev->GetMac()->TraceConnectWithoutContext("TxOkHeader",MakeBoundCallback(&Scenario::Ok,this,i));
      PointerValue pointer;dev->GetPhy()->GetAttribute("State",pointer);Require(pointer.Get<Object>()->TraceConnectWithoutContext("State",MakeBoundCallback(&Scenario::StateLog,this,i)),"PHY state trace");
      Require(dev->GetPhy()->TraceConnectWithoutContext("PhyRxDrop",MakeBoundCallback(&Scenario::DropLog,this,i)),"PHY drop trace");
      dev->GetMac()->GetAttribute("Txop",pointer);auto txop=pointer.Get<Txop>();Require(bool(txop),"native DCF Txop");
      Require(txop->TraceConnectWithoutContext("BackoffTrace",MakeBoundCallback(&Scenario::BackoffLog,this,i)),"backoff trace");Require(txop->TraceConnectWithoutContext("CwTrace",MakeBoundCallback(&Scenario::CwLog,this,i)),"CW trace");
      sockets[i]=Socket::CreateSocket(nodes.Get(i),UdpSocketFactory::GetTypeId());sockets[i]->SetAllowBroadcast(true);sockets[i]->SetIpTtl(1);Require(sockets[i]->Bind(InetSocketAddress(Ipv4Address::GetAny(),9000))==0,"UDP bind");sockets[i]->SetRecvCallback(MakeBoundCallback(&Scenario::Rx,this,i));}
    if(!summaryOnly){tx.open(output+".tx.csv");plant.open(output+".plant.csv");deliveries.open(output+".delivery.csv");events.open(output+".events.csv");receives.open(output+".receives.csv");macok.open(output+".mac_ok.csv");for(auto f:{&tx,&plant,&deliveries,&events,&receives,&macok}){Require(bool(*f),"log file unavailable");*f<<std::setprecision(17);}
      tx<<"start_s,end_s,node,kind,frame_bytes,uid,sequence,peer\n";plant<<"time_s,node,error_m,state_generation_s,estimate_position_error_m,estimate_velocity_error_m_s,x,y,z,vx,vy,vz\n";deliveries<<"arrival_s,id,destination,generated_s,hops\n";events<<"time_s,event,id,destination,hops\n";receives<<"time_s,node,kind,id,uid\n";macok<<"time_s,node,peer,sequence\n";}
    Simulator::Schedule(MicroSeconds(1),&Scenario::Plant,this);for(uint32_t i=0;i<n;i++)Simulator::Schedule(Seconds(double(i)/n),&Scenario::Hello,this,i);Simulator::Schedule(Seconds(3),&Scenario::Offer,this,n-1);Simulator::Schedule(Seconds(3.08),&Scenario::Offer,this,n-2);if(outage)Simulator::Schedule(Seconds(10),&Scenario::Outage,this);}
  static double Mean(std::vector<double>v){double x=0;for(auto a:v)x+=a;return v.empty()?0:x/v.size();}
  static double Quantile(std::vector<double>v,double q){if(v.empty())return 0;std::sort(v.begin(),v.end());double i=(v.size()-1)*q;int lo=i;return v[lo]+(i-lo)*(v[std::min(size_t(lo+1),v.size()-1)]-v[lo]);}
  void Finish(){
    std::ofstream f(output+".json");Require(bool(f),"summary unavailable");
    std::string autonomousModel=absoluteSpeed<=1e-12?"ns3::ConstantPositionMobilityModel":(mission=="waypoint"?"ns3::RandomWaypointMobilityModel":(mission=="gauss_markov"?"ns3::GaussMarkovMobilityModel":"ns3::ConstantVelocityMobilityModel"));
    std::string followerModel=absoluteSpeed<=1e-12?"ns3::ConstantPositionMobilityModel":"ns3::ConstantVelocityMobilityModel";
    uint64_t completed=macOkHeaders+failures;
    uint64_t retries=unicastPhyAttempts>completed?unicastPhyAttempts-completed:0;
    uint64_t acceptedUpdates=0,rejectedUpdates=0;for(auto d:destinations){acceptedUpdates+=filters[d].accepted;rejectedUpdates+=filters[d].rejected;}
    f<<std::setprecision(17)
     <<"{\n\"engine\":\"native_ns3_3.32_relative_speed_v5\",\n"
     <<"\"method\":\""<<method<<"\",\n\"mission\":\""<<mission<<"\",\n\"estimator\":\""<<estimator<<"\",\n"
     <<"\"adaptation_band\":\""<<adaptationBand<<"\",\n"
     <<"\"native_mobility_model\":\""<<autonomousModel<<"\",\n"
     <<"\"native_mobility_model_by_role\":{\"source_and_ordinary_relays\":\""<<autonomousModel<<"\",\"controlled_followers\":\""<<followerModel<<"\"},\n"
     <<"\"mobility_only\":"<<(mobilityOnly?"true":"false")<<",\n"
     <<"\"absolute_speed_target_m_s\":"<<absoluteSpeed<<",\n"
     <<"\"formation_relative_speed_target_m_s\":"<<(mission=="formation"?std::to_string(formationRelativeSpeedTarget):"null")<<",\n"
     <<"\"formation_excitation_period_s\":"<<(mission=="formation"?std::to_string(formationExcitationPeriod):"null")<<",\n"
     <<"\"formation_excitation_sharpness\":"<<(mission=="formation"?std::to_string(formationExcitationSharpness):"null")<<",\n"
     <<"\"mean_actual_absolute_speed_m_s\":"<<Mean(absoluteSpeedSamples)<<",\n"
     <<"\"p95_actual_absolute_speed_m_s\":"<<Quantile(absoluteSpeedSamples,.95)<<",\n"
     <<"\"maximum_absolute_speed_error_m_s\":"<<maximumAbsoluteSpeedError<<",\n"
     <<"\"mean_connected_pair_relative_speed_m_s\":"<<Mean(relativeSpeedSamples)<<",\n"
     <<"\"p95_connected_pair_relative_speed_m_s\":"<<Quantile(relativeSpeedSamples,.95)<<",\n"
     <<"\"position_measurement_std_m\":"<<positionMeasurementStd<<",\n\"velocity_measurement_std_m_s\":"<<velocityMeasurementStd<<",\n\"process_acceleration_std_m_s2\":"<<processAccelerationStd<<",\n"
     <<"\"observation_noise_scalars\":"<<observationNoiseScalars<<",\n\"observation_noise_checksum\":"<<observationNoiseChecksum<<",\n"
     <<"\"seed\":"<<seed<<",\n\"nodes\":"<<n<<",\n"
     <<"\"summary_only\":"<<(summaryOnly?"true":"false")<<",\n"
     <<"\"mute\":"<<(mute?"true":"false")<<",\n"
     <<"\"offered\":"<<offered<<",\n\"delivered\":"<<delivered<<",\n"
     <<"\"pdr\":"<<(offered?double(delivered)/offered:0)<<",\n"
     <<"\"tracking_rmse_m\":"<<(errors?std::sqrt(squaredError/errors):0)<<",\n"
     <<"\"estimator_position_rmse_m\":"<<(estimateErrors?std::sqrt(squaredPositionEstimateError/estimateErrors):0)<<",\n"
     <<"\"estimator_velocity_rmse_m_s\":"<<(estimateErrors?std::sqrt(squaredVelocityEstimateError/estimateErrors):0)<<",\n"
     <<"\"mean_estimator_covariance_trace\":"<<(estimator=="kf"&&estimateErrors?summedCovarianceTrace/estimateErrors:0)<<",\n"
     <<"\"kalman_accepted_updates\":"<<(estimator=="kf"?acceptedUpdates:0)<<",\n\"kalman_rejected_stale_updates\":"<<(estimator=="kf"?rejectedUpdates:0)<<",\n"
     <<"\"mean_delay_s\":"<<Mean(delays)<<",\n\"p95_delay_s\":"<<Quantile(delays,.95)<<",\n"
     <<"\"mean_state_age_s\":"<<Mean(ages)<<",\n\"p95_state_age_s\":"<<Quantile(ages,.95)<<",\n"
     <<"\"minimum_delivered_hops\":"<<(delivered?std::to_string(minHops):"null")<<",\n"
     <<"\"upper_control_bytes\":"<<upperControlBytes<<",\n"
     <<"\"phy_bytes\":"<<phyBytes<<",\n\"phy_control_bytes\":"<<phyControlBytes<<",\n"
     <<"\"native_ack_bytes\":"<<ackBytes<<",\n\"native_ack_frames\":"<<ackFrames<<",\n"
     <<"\"phy_frames\":"<<phyFrames<<",\n\"summed_tx_airtime_s\":"<<summedTxAirtime<<",\n"
     <<"\"unicast_phy_attempts\":"<<unicastPhyAttempts<<",\n\"group_phy_attempts\":"<<groupPhyAttempts<<",\n"
     <<"\"mac_tx_successes\":"<<macOkHeaders<<",\n\"estimated_retries\":"<<retries<<",\n"
     <<"\"estimated_retry_fraction\":"<<(unicastPhyAttempts?double(retries)/unicastPhyAttempts:0)<<",\n"
     <<"\"final_unicast_failures\":"<<failures<<",\n"
     <<"\"final_unicast_failure_rate\":"<<(completed?double(failures)/completed:0)<<",\n"
     <<"\"rx_drop_events\":"<<rxDropEvents<<",\n"
     <<"\"backoff_draws\":"<<backoffDraws<<",\n\"backoff_slots\":"<<backoffSlots<<",\n"
     <<"\"mean_backoff_slots\":"<<(backoffDraws?double(backoffSlots)/backoffDraws:0)<<",\n"
     <<"\"cw_changes\":"<<cwChanges<<",\n\"max_cw_observed\":"<<maxCwObserved<<",\n"
     <<"\"rerr_receives\":"<<rerrRx<<",\n\"feature_observations\":"<<observations<<",\n"
     <<"\"kind_frame_bytes\":[";
    for(int k=0;k<8;k++)f<<(k?",":"")<<kindBytes[k];
    f<<"],\n\"kind_frames\":[";for(int k=0;k<8;k++)f<<(k?",":"")<<kindFrames[k];
    f<<"],\n\"phy_state_seconds\":{";bool first=true;
    for(auto &x:phyStateSeconds){f<<(first?"":",")<<"\""<<x.first<<"\":"<<x.second;first=false;}
    f<<"},\n\"rx_drop_reason_counts\":{";first=true;
    for(auto &x:rxDropReasons){f<<(first?"":",")<<"\""<<x.first<<"\":"<<x.second;first=false;}
    f<<"}\n}\n";
  }
};
int main(int argc,char**argv){try{Scenario s;bool parity=false,kalmanParity=false;CommandLine cmd;cmd.AddValue("method","hop, analytical, frozen, pooled, family_adapted",s.method);cmd.AddValue("mission","waypoint, gauss_markov, formation",s.mission);cmd.AddValue("adaptationBand","low, medium, or high for family_adapted",s.adaptationBand);cmd.AddValue("speed","actual absolute speed in metres per second",s.absoluteSpeed);cmd.AddValue("formationRelativeSpeedTarget","preregistered formation relative-speed excitation in metres per second",s.formationRelativeSpeedTarget);cmd.AddValue("formationExcitationPeriod","smooth formation heading-reversal period in seconds",s.formationExcitationPeriod);cmd.AddValue("formationExcitationSharpness","dimensionless smooth formation heading-reversal sharpness",s.formationExcitationSharpness);cmd.AddValue("nodes","node count",s.n);cmd.AddValue("seed","whole-task seed",s.seed);cmd.AddValue("input","input directory",s.input);cmd.AddValue("output","output file prefix",s.output);cmd.AddValue("estimator","cv or kf",s.estimator);cmd.AddValue("positionMeasurementStd","position observation standard deviation in metres",s.positionMeasurementStd);cmd.AddValue("velocityMeasurementStd","velocity observation standard deviation in metres per second",s.velocityMeasurementStd);cmd.AddValue("processAccelerationStd","KF white-acceleration standard deviation in metres per second squared",s.processAccelerationStd);cmd.AddValue("mute","block delivered controller updates only",s.mute);cmd.AddValue("outage","five-second relay radio outage diagnostic",s.outage);cmd.AddValue("summaryOnly","write aggregate JSON without large CSV traces",s.summaryOnly);cmd.AddValue("mobilityOnly","write native mobility states without radio traffic",s.mobilityOnly);cmd.AddValue("parity","score parity file only",parity);cmd.AddValue("kalmanParity","deterministic KF parity sequence only",kalmanParity);cmd.Parse(argc,argv);
  if(parity){std::ifstream f(s.input+"/parity_cases.dat");Require(bool(f),"parity input");std::ofstream o(s.output+".parity.csv");uint32_t count;f>>count;o<<"case,model,prediction\n"<<std::setprecision(17);
    for(uint32_t c=0;c<count;c++){std::string path;int dim;f>>path>>dim;std::vector<double>x(dim);for(auto &v:x)f>>v;Fit fit;fit.Read(s.input+"/"+path);o<<c<<","<<path<<","<<fit.Predict(x)<<"\n";}
    std::ifstream features(s.input+"/feature_cases.dat");Require(bool(features),"feature cases");features>>count;std::ofstream fo(s.output+".features.csv");fo<<"case,f0,f1,f2,f3,f4,swap_error\n"<<std::setprecision(17);
    for(uint32_t i=0;i<count;i++){State a,b;for(auto &v:a.p)features>>v;for(auto &v:a.v)features>>v;for(auto &v:b.p)features>>v;for(auto &v:b.v)features>>v;auto x=Features(a,b),y=Features(b,a);double error=0;fo<<i;for(size_t j=0;j<x.size();j++){fo<<","<<x[j];error=std::max(error,std::abs(x[j]-y[j]));}fo<<","<<error<<"\n";}
    return 0;}
  if(kalmanParity){std::ifstream f(s.input+"/kf_parity_case.dat");Require(bool(f),"KF parity input");State initial;double positionStd,velocityStd,accelerationStd;uint32_t count;for(auto &x:initial.p)f>>x;for(auto &x:initial.v)f>>x;f>>initial.time>>positionStd>>velocityStd>>accelerationStd>>count;IntermittentCvKalman filter;filter.Initialize(initial,positionStd,velocityStd,accelerationStd);std::ofstream o(s.output+".kf.csv");Require(bool(o),"KF parity output");o<<"step,px,py,pz,vx,vy,vz,p00,p01,p11,last_measurement,accepted,rejected\n"<<std::setprecision(17);for(uint32_t step=0;step<count;step++){State measurement;double reception,snapshot;for(auto &x:measurement.p)f>>x;for(auto &x:measurement.v)f>>x;f>>measurement.time>>reception>>snapshot;Require(bool(f),"short KF parity input");filter.UpdateDelayed(measurement,reception);auto state=filter.Snapshot(snapshot);o<<step;for(auto x:state.p)o<<","<<x;for(auto x:state.v)o<<","<<x;o<<","<<filter.covariance[0][0]<<","<<filter.covariance[0][1]<<","<<filter.covariance[1][1]<<","<<filter.lastMeasurementTime<<","<<filter.accepted<<","<<filter.rejected<<"\n";}return 0;}
  Require(s.seed>=1,"seed index");Require(s.mission=="waypoint"||s.mission=="gauss_markov"||s.mission=="formation","mission");Require(s.estimator=="cv"||s.estimator=="kf","estimator");Require(std::min({s.positionMeasurementStd,s.velocityMeasurementStd,s.processAccelerationStd})>=0,"negative estimator standard deviation");Require(s.method=="hop"||s.method=="analytical"||s.method=="frozen"||s.method=="pooled"||s.method=="family_adapted","method");if(s.method=="family_adapted")Require(s.adaptationBand=="low"||s.adaptationBand=="medium"||s.adaptationBand=="high","family-adapted method requires a declared relative-speed band");RngSeedManager::SetSeed(1729);RngSeedManager::SetRun(s.seed);s.Setup();Simulator::Stop(Seconds(40));Simulator::Run();s.Finish();Simulator::Destroy();std::cout<<"P2_NATIVE_RELATIVE_SPEED_V5_FINISHED "<<s.output<<"\n";return 0;
}catch(const std::exception&e){std::cerr<<"P2_NATIVE_FAILURE "<<e.what()<<"\n";return 2;}}
