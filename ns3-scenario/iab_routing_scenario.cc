#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/applications-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/error-model.h"
#include "ns3/pointer.h"
#include <fstream>
#include <random>
#include <vector>
#include <tuple>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("IabRoutingScenario");

struct LinkSpec {
    uint32_t src, dst;
    double delay_ms;
    double pb;
    double capacity_mbps;
};

int main(int argc, char *argv[])
{
    double simTime = 20.0;
    uint32_t seed = 42;
    std::string topoOut = "topology.csv";
    std::string flowOut = "flows.csv";

    CommandLine cmd;
    cmd.AddValue("simTime", "Simulation time (s)", simTime);
    cmd.AddValue("seed", "RNG seed for random pb per link", seed);
    cmd.AddValue("topoOut", "Path to write topology.csv", topoOut);
    cmd.AddValue("flowOut", "Path to write flows.csv", flowOut);
    cmd.Parse(argc, argv);

    std::mt19937 rng(seed);
    // Block error rate from the 3GPP mmWave link budget rather than a uniform draw.
    // TR 38.901 Urban Micro at 28 GHz: PL = 32.4 + 21*log10(d) + 20*log10(f_GHz) for
    // LOS, plus a 20 dB penalty when the link is blocked. SINR follows from 23 dBm TX
    // power, 100 MHz bandwidth and a 7 dB noise figure; the BLER curve is the standard
    // sigmoid around the MCS threshold. Distances come from the hexagonal layout in
    // Fig. 4: the donor at the centre, IAB nodes on 200 m rings.
    std::uniform_real_distribution<double> blockDist(0.0, 1.0);

    auto pathlossDb = [](double distM, bool los) {
        double pl = 32.4 + 21.0 * std::log10(std::max(distM, 1.0)) + 20.0 * std::log10(28.0);
        return los ? pl : pl + 20.0;   // NLOS penalty from blockage
    };

    auto blerFromSinr = [](double sinrDb) {
        // sigmoid BLER curve: ~0.1 at the MCS threshold, falling sharply above it
        const double threshold = 12.0;   // dB, for the MCS the scheduler picks
        return 1.0 / (1.0 + std::exp(1.2 * (sinrDb - threshold)));
    };

    const double txPowerDbm = 23.0;
    // Beamforming gain from the phased arrays at both ends. Without it no 28 GHz link
    // closes at 200 m: the budget above gives 0.3 dB SINR, well under the MCS threshold,
    // and every link would sit at the BLER ceiling. 12 dBi at the IAB node and 10 dBi at
    // the UE is a conventional array pair for this band.
    const double beamGainDb = 22.0;
    const double noiseFigureDb = 7.0;
    const double bandwidthHz = 100e6;
    const double thermalNoiseDbm = -174.0 + 10.0 * std::log10(bandwidthHz) + noiseFigureDb;

    std::uniform_real_distribution<double> pbDist(0.001, 0.05);
    std::uniform_real_distribution<double> delayJitter(0.8, 1.2);

    const uint32_t DONOR = 0;
    const uint32_t NUM_IAB = 18;
    const uint32_t NUM_UE_PER_IAB = 2;
    const uint32_t FIRST_IAB = 1;
    const uint32_t FIRST_UE = FIRST_IAB + NUM_IAB;
    const uint32_t NUM_UE = NUM_IAB * NUM_UE_PER_IAB;
    const uint32_t TOTAL_NODES = FIRST_UE + NUM_UE;

    NodeContainer nodes;
    nodes.Create(TOTAL_NODES);

    std::vector<LinkSpec> links;

    auto addLink = [&](uint32_t a, uint32_t b, double baseDelay) {
        double d = baseDelay * delayJitter(rng);
        // Hexagonal layout of Fig. 4: IAB nodes sit on 200 m rings around the donor,
        // UE access links are shorter. The stored delay is the scheduling delay, not
        // propagation, so distance comes from the layout rather than from it.
        bool accessLink = (a >= 19 || b >= 19);
        double distM = accessLink ? 80.0 : 200.0;
        bool los = blockDist(rng) > 0.2;          // 20% of links blocked at any time
        double rxDbm = txPowerDbm + beamGainDb - pathlossDb(distM, los);
        double sinrDb = rxDbm - thermalNoiseDbm;
        double pb = std::min(std::max(blerFromSinr(sinrDb), 1e-4), 0.5);
        links.push_back({a, b, d, pb, 1000.0});
    };

    for (uint32_t g = 0; g < 3; g++) {
        uint32_t base = FIRST_IAB + g * 6;
        for (uint32_t i = 0; i < 6; i++) {
            uint32_t a = base + i;
            uint32_t b = base + (i + 1) % 6;
            addLink(a, b, 3.0);
        }
        addLink(DONOR, base, 5.0);
    }
    addLink(FIRST_IAB, FIRST_IAB + 6, 4.0);
    addLink(FIRST_IAB + 6, FIRST_IAB + 12, 4.0);
    addLink(FIRST_IAB + 12, FIRST_IAB, 4.0);

    for (uint32_t n = 0; n < NUM_IAB; n++) {
        uint32_t iabId = FIRST_IAB + n;
        for (uint32_t u = 0; u < NUM_UE_PER_IAB; u++) {
            uint32_t ueId = FIRST_UE + n * NUM_UE_PER_IAB + u;
            addLink(iabId, ueId, 1.0);
        }
    }

    InternetStackHelper internet;
    internet.Install(nodes);

    PointToPointHelper p2p;
    Ipv4AddressHelper address;
    int subnet = 0;

    for (auto &l : links) {
        p2p.SetDeviceAttribute("DataRate", StringValue("1000Mbps"));
        p2p.SetChannelAttribute("Delay", TimeValue(MilliSeconds(l.delay_ms)));
        NetDeviceContainer dev = p2p.Install(nodes.Get(l.src), nodes.Get(l.dst));

        Ptr<RateErrorModel> em = CreateObject<RateErrorModel>();
        em->SetAttribute("ErrorRate", DoubleValue(l.pb));
        em->SetAttribute("ErrorUnit", StringValue("ERROR_UNIT_PACKET"));
        dev.Get(0)->SetAttribute("ReceiveErrorModel", PointerValue(em));
        Ptr<RateErrorModel> em2 = CreateObject<RateErrorModel>();
        em2->SetAttribute("ErrorRate", DoubleValue(l.pb));
        em2->SetAttribute("ErrorUnit", StringValue("ERROR_UNIT_PACKET"));
        dev.Get(1)->SetAttribute("ReceiveErrorModel", PointerValue(em2));

        std::ostringstream base;
        base << "10." << (subnet / 254) << "." << (subnet % 254) << ".0";
        address.SetBase(base.str().c_str(), "255.255.255.0");
        address.Assign(dev);
        subnet++;
    }

    Ipv4GlobalRoutingHelper::PopulateRoutingTables();

    {
        std::ofstream out(topoOut);
        out << "src,dst,delay_ms,pb,capacity_mbps\n";
        for (auto &l : links) {
            out << l.src << "," << l.dst << "," << l.delay_ms << ","
                << l.pb << "," << l.capacity_mbps << "\n";
        }
        out.close();
        std::cout << "Wrote " << topoOut << " (" << links.size() << " links)" << std::endl;
    }

    uint16_t port = 9000;
    UdpServerHelper server(port);
    ApplicationContainer serverApp = server.Install(nodes.Get(DONOR));
    serverApp.Start(Seconds(0.0));
    serverApp.Stop(Seconds(simTime));

    Ipv4Address donorAddr = nodes.Get(DONOR)->GetObject<Ipv4>()->GetAddress(1, 0).GetLocal();

    ApplicationContainer clientApps;
    for (uint32_t u = 0; u < NUM_UE; u++) {
        UdpClientHelper client(donorAddr, port);
        client.SetAttribute("MaxPackets", UintegerValue(100000));
        client.SetAttribute("Interval", TimeValue(MilliSeconds(30)));
        client.SetAttribute("PacketSize", UintegerValue(1024));
        ApplicationContainer c = client.Install(nodes.Get(FIRST_UE + u));
        c.Start(Seconds(1.0 + 0.01 * u));
        c.Stop(Seconds(simTime));
        clientApps.Add(c);
    }

    FlowMonitorHelper flowmon;
    Ptr<FlowMonitor> monitor = flowmon.InstallAll();

    Simulator::Stop(Seconds(simTime));
    Simulator::Run();

    monitor->CheckForLostPackets();
    Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier>(flowmon.GetClassifier());
    std::map<FlowId, FlowMonitor::FlowStats> stats = monitor->GetFlowStats();

    std::ofstream out(flowOut);
    out << "flow_id,src,dst,tx_packets,rx_packets,lost_packets,mean_delay_ms,throughput_mbps,empirical_reliability\n";
    for (auto &s : stats) {
        Ipv4FlowClassifier::FiveTuple t = classifier->FindFlow(s.first);
        double meanDelay = s.second.rxPackets > 0
            ? (s.second.delaySum.GetSeconds() / s.second.rxPackets) * 1000.0 : 0.0;
        double throughput = s.second.rxBytes * 8.0 / simTime / 1e6;
        double reliability = s.second.txPackets > 0
            ? static_cast<double>(s.second.rxPackets) / s.second.txPackets : 0.0;
        out << s.first << "," << t.sourceAddress << "," << t.destinationAddress << ","
            << s.second.txPackets << "," << s.second.rxPackets << "," << s.second.lostPackets << ","
            << meanDelay << "," << throughput << "," << reliability << "\n";
    }
    out.close();
    std::cout << "Wrote " << flowOut << " (" << stats.size() << " flows)" << std::endl;

    Simulator::Destroy();
    return 0;
}
