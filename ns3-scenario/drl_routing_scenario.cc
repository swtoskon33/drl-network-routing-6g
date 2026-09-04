// Abilene-like 11-node topology, variable UDP traffic, per-link delay/utilisation
// exported to CSV via FlowMonitor. This is the ground-truth environment that
// both Dijkstra and the DRL agent will be scored against.
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/applications-module.h"
#include "ns3/flow-monitor-module.h"
#include <fstream>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("DrlRoutingScenario");

int main(int argc, char *argv[])
{
    uint32_t numNodes = 11;
    double simTime = 30.0;
    std::string traceOut = "traces/run_default.csv";

    CommandLine cmd;
    cmd.AddValue("simTime", "Simulation time (s)", simTime);
    cmd.AddValue("traceOut", "CSV output path", traceOut);
    cmd.Parse(argc, argv);

    NodeContainer nodes;
    nodes.Create(numNodes);

    // Abilene-ish edges: {src, dst, delay_ms}
    std::vector<std::tuple<int,int,double>> edges = {
        {0,1,6},{1,2,8},{2,3,5},{3,4,7},{4,5,6},
        {5,6,9},{6,7,4},{7,8,6},{8,9,5},{9,10,7},
        {0,4,12},{1,6,11},{2,8,13},{3,9,10},{5,10,8}
    };

    InternetStackHelper internet;
    internet.Install(nodes);

    PointToPointHelper p2p;
    p2p.SetDeviceAttribute("DataRate", StringValue("1000Mbps"));

    Ipv4AddressHelper address;
    int subnet = 0;
    std::vector<NetDeviceContainer> devs;

    for (auto &e : edges) {
        int a = std::get<0>(e), b = std::get<1>(e);
        double delay = std::get<2>(e);
        p2p.SetChannelAttribute("Delay", TimeValue(MilliSeconds(delay)));
        NetDeviceContainer d = p2p.Install(nodes.Get(a), nodes.Get(b));
        std::ostringstream base;
        base << "10." << (subnet / 254) << "." << (subnet % 254) << ".0";
        address.SetBase(base.str().c_str(), "255.255.255.0");
        address.Assign(d);
        devs.push_back(d);
        subnet++;
    }

    Ipv4GlobalRoutingHelper::PopulateRoutingTables();

    // Variable UDP traffic between random node pairs (stand-in for 6G user traffic)
    uint16_t port = 9000;
    for (int i = 0; i < 6; i++) {
        int src = rand() % numNodes;
        int dst = rand() % numNodes;
        if (src == dst) continue;

        UdpServerHelper server(port + i);
        ApplicationContainer serverApp = server.Install(nodes.Get(dst));
        serverApp.Start(Seconds(0.0));
        serverApp.Stop(Seconds(simTime));

        UdpClientHelper client(nodes.Get(dst)->GetObject<Ipv4>()->GetAddress(1,0).GetLocal(), port + i);
        client.SetAttribute("MaxPackets", UintegerValue(100000));
        client.SetAttribute("Interval", TimeValue(MilliSeconds(10)));
        client.SetAttribute("PacketSize", UintegerValue(1024));
        ApplicationContainer clientApp = client.Install(nodes.Get(src));
        clientApp.Start(Seconds(1.0));
        clientApp.Stop(Seconds(simTime));
    }

    FlowMonitorHelper flowmon;
    Ptr<FlowMonitor> monitor = flowmon.InstallAll();

    Simulator::Stop(Seconds(simTime));
    Simulator::Run();

    monitor->CheckForLostPackets();
    Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier>(flowmon.GetClassifier());
    std::map<FlowId, FlowMonitor::FlowStats> stats = monitor->GetFlowStats();

    std::ofstream out(traceOut);
    out << "flow_id,src,dst,tx_packets,rx_packets,mean_delay_ms,throughput_mbps\n";
    for (auto &s : stats) {
        Ipv4FlowClassifier::FiveTuple t = classifier->FindFlow(s.first);
        double meanDelay = s.second.rxPackets > 0
            ? (s.second.delaySum.GetSeconds() / s.second.rxPackets) * 1000.0 : 0.0;
        double throughput = s.second.rxBytes * 8.0 / simTime / 1e6;
        out << s.first << "," << t.sourceAddress << "," << t.destinationAddress << ","
            << s.second.txPackets << "," << s.second.rxPackets << ","
            << meanDelay << "," << throughput << "\n";
    }
    out.close();

    Simulator::Destroy();
    std::cout << "Wrote " << traceOut << std::endl;
    return 0;
}
