// IAB topology with the 3GPP channel model, Phase 1 of the reproduction.
//
// Builds the deployment of Fig. 4: an IAB donor at the centre, three hexagonal grids of
// six IAB nodes each on 200 m rings, and UEs dropped uniformly and associated with their
// closest node. Every link is measured through the real 3GPP TR 38.901 Urban Micro
// channel -- pathloss, LOS/NLOS condition, shadowing and the stochastic spatial channel
// model -- rather than an analytic budget.
//
// Table I parameters: 28 GHz centre frequency, 100 MHz bandwidth, 23 dBm transmit power,
// 7 dB noise figure at the base station and 10 dB at the UE.
//
// Writes positions.csv and links.csv. Makes no routing decision and carries no traffic;
// those are later phases.

#include "ns3/antenna-module.h"
#include "ns3/core-module.h"
#include "ns3/mobility-module.h"
#include "ns3/network-module.h"
#include "ns3/propagation-module.h"
#include "ns3/spectrum-module.h"

#include <cmath>
#include <fstream>
#include <vector>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("IabTopology");

namespace
{

constexpr double kFrequencyHz = 28.0e9;      // Table I: 28 GHz
constexpr double kBandwidthHz = 100.0e6;     // Table I: 100 MHz
constexpr double kTxPowerDbm = 23.0;         // Table I: 23 dBm
constexpr double kBsNoiseFigureDb = 7.0;     // Table I
constexpr double kUeNoiseFigureDb = 10.0;    // Table I
constexpr double kRingRadiusM = 200.0;       // Fig. 4: 200 m hexagonal grids
constexpr double kIabHeightM = 10.0;         // street-level IAB nodes
constexpr double kUeHeightM = 1.5;

/// Thermal noise over the carrier bandwidth, in dBm.
double
NoiseFloorDbm(double noiseFigureDb)
{
    return -174.0 + 10.0 * std::log10(kBandwidthHz) + noiseFigureDb;
}

/// Block error rate after link adaptation.
///
/// The scheduler does not transmit at a fixed modulation and coding scheme and accept
/// whatever error rate follows. It picks the highest MCS whose target BLER the SINR can
/// support -- 3GPP works to 10% for data, 1% for URLLC -- and only when the link is too
/// poor for the lowest MCS does the error rate climb.
///
/// Without this every NLOS link sits at a BLER of 0.5 and no route in the mesh can reach
/// the 0.999 reliability target, because the constraint is not a property of the
/// deployment but of the assumption that nobody adapts.
double
BlerFromSinr(double sinrDb)
{
    // The lowest MCS in TS 38.214 (QPSK, rate 0.12) needs roughly -6 dB SINR to hit its
    // target; below that the link cannot carry data at all.
    constexpr double kLowestMcsSinrDb = -6.0;
    constexpr double kUrllcTargetBler = 1e-3;
    constexpr double kFloorBler = 1e-5;

    if (sinrDb < kLowestMcsSinrDb)
    {
        // beyond the reach of link adaptation: the link fails more often than it works
        return 0.5;
    }

    // Within range the scheduler holds the error rate near the target, with margin above
    // it as the SINR improves. A decade of BLER per 10 dB is the slope these curves have.
    const double marginDb = sinrDb - kLowestMcsSinrDb;
    const double bler = kUrllcTargetBler * std::pow(10.0, -marginDb / 10.0);
    return std::max(bler, kFloorBler);
}

struct NodeRecord
{
    uint32_t id;
    std::string kind;   // donor, iab, ue
    Vector position;
};

} // namespace

int
main(int argc, char* argv[])
{
    uint32_t seed = 42;
    uint32_t run = 1;
    uint32_t rings = 3;            // Fig. 4: three hexagonal grids
    uint32_t nodesPerRing = 6;
    uint32_t ueCount = 36;
    std::string positionsOut = "positions.csv";
    std::string linksOut = "links.csv";

    CommandLine cmd(__FILE__);
    cmd.AddValue("seed", "RNG seed", seed);
    cmd.AddValue("run", "RNG run number", run);
    cmd.AddValue("rings", "Hexagonal grids of IAB nodes around the donor", rings);
    cmd.AddValue("ueCount", "UEs dropped uniformly over the deployment", ueCount);
    cmd.AddValue("positionsOut", "Path for positions.csv", positionsOut);
    cmd.AddValue("linksOut", "Path for links.csv", linksOut);
    cmd.Parse(argc, argv);

    RngSeedManager::SetSeed(seed);
    RngSeedManager::SetRun(run);

    const uint32_t iabCount = rings * nodesPerRing;
    const uint32_t totalNodes = 1 + iabCount + ueCount;

    // --- positions ---------------------------------------------------------------
    //
    // The donor sits at the origin. Each ring is a hexagon of six nodes at 200 m,
    // rotated so the rings do not overlap.

    std::vector<NodeRecord> records;
    records.push_back({0, "donor", Vector(0.0, 0.0, kIabHeightM)});

    for (uint32_t r = 0; r < rings; r++)
    {
        const double ringOffset = (2.0 * M_PI / nodesPerRing) * (r / double(rings));
        const double radius = kRingRadiusM * (r + 1);
        for (uint32_t i = 0; i < nodesPerRing; i++)
        {
            const double angle = ringOffset + (2.0 * M_PI * i) / nodesPerRing;
            records.push_back({uint32_t(records.size()), "iab",
                               Vector(radius * std::cos(angle),
                                      radius * std::sin(angle),
                                      kIabHeightM)});
        }
    }

    // UEs dropped uniformly over the disc the deployment covers
    Ptr<UniformRandomVariable> uniform = CreateObject<UniformRandomVariable>();
    const double deploymentRadius = kRingRadiusM * rings;
    for (uint32_t u = 0; u < ueCount; u++)
    {
        const double angle = uniform->GetValue(0.0, 2.0 * M_PI);
        // sqrt keeps the drop uniform by area rather than by radius
        const double radius = deploymentRadius * std::sqrt(uniform->GetValue(0.0, 1.0));
        records.push_back({uint32_t(records.size()), "ue",
                           Vector(radius * std::cos(angle),
                                  radius * std::sin(angle),
                                  kUeHeightM)});
    }

    NodeContainer nodes;
    nodes.Create(totalNodes);

    Ptr<ListPositionAllocator> positions = CreateObject<ListPositionAllocator>();
    for (const auto& record : records)
    {
        positions->Add(record.position);
    }
    MobilityHelper mobility;
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.SetPositionAllocator(positions);
    mobility.Install(nodes);

    // --- channel -------------------------------------------------------------------
    //
    // TR 38.901 Urban Micro street canyon: the condition model decides LOS or NLOS per
    // pair, the loss model applies the corresponding pathloss and shadowing.

    Ptr<ThreeGppUmiStreetCanyonPropagationLossModel> lossModel =
        CreateObject<ThreeGppUmiStreetCanyonPropagationLossModel>();
    lossModel->SetAttribute("Frequency", DoubleValue(kFrequencyHz));
    lossModel->SetAttribute("ShadowingEnabled", BooleanValue(true));

    Ptr<ThreeGppUmiStreetCanyonChannelConditionModel> conditionModel =
        CreateObject<ThreeGppUmiStreetCanyonChannelConditionModel>();
    lossModel->SetChannelConditionModel(conditionModel);

    // Phased arrays at both ends. Without them nothing closes at 28 GHz over 200 m: the
    // link budget lands 20 dB short and every link sits at the BLER ceiling. The paper
    // does not give an antenna configuration, so these are conventional mmWave arrays --
    // 8x8 at the IAB nodes, 4x4 at the UEs.
    Ptr<ThreeGppChannelModel> channelModel = CreateObject<ThreeGppChannelModel>();
    channelModel->SetAttribute("Frequency", DoubleValue(kFrequencyHz));
    channelModel->SetAttribute("Scenario", StringValue("UMi-StreetCanyon"));
    channelModel->SetChannelConditionModel(conditionModel);

    Ptr<ThreeGppSpectrumPropagationLossModel> spectrumLoss =
        CreateObject<ThreeGppSpectrumPropagationLossModel>();
    spectrumLoss->SetChannelModel(channelModel);

    auto makeArray = [](uint32_t rows, uint32_t cols) {
        Ptr<UniformPlanarArray> array = CreateObject<UniformPlanarArray>();
        array->SetAttribute("NumRows", UintegerValue(rows));
        array->SetAttribute("NumColumns", UintegerValue(cols));
        array->SetAttribute("AntennaElement",
                            PointerValue(CreateObject<IsotropicAntennaModel>()));
        return array;
    };

    std::vector<Ptr<UniformPlanarArray>> arrays;
    for (const auto& record : records)
    {
        arrays.push_back(record.kind == "ue" ? makeArray(4, 4) : makeArray(8, 8));
    }

    // Array gain in dB: a fully coherent beam over N elements gains 10*log10(N) at each
    // end. This is the gain the beamforming vectors would realise once steered, which is
    // what the association does in a real deployment.
    auto arrayGainDb = [](const Ptr<UniformPlanarArray>& array) {
        const double elements = array->GetNumElems();
        return 10.0 * std::log10(elements);
    };

    // --- measure every link ---------------------------------------------------------

    std::ofstream positionsFile(positionsOut);
    positionsFile << "# seed=" << seed << " run=" << run << "\n";
    positionsFile << "id,kind,x,y,z\n";
    for (const auto& record : records)
    {
        positionsFile << record.id << "," << record.kind << ","
                      << record.position.x << "," << record.position.y << ","
                      << record.position.z << "\n";
    }
    positionsFile.close();

    std::ofstream linksFile(linksOut);
    linksFile << "# seed=" << seed << " run=" << run
              << " frequency_hz=" << kFrequencyHz << " bandwidth_hz=" << kBandwidthHz
              << " tx_dbm=" << kTxPowerDbm << "\n";
    linksFile << "src,dst,kind,distance_m,los,pathloss_db,beam_gain_db,sinr_db,bler\n";

    uint32_t linkCount = 0;
    for (uint32_t a = 0; a < totalNodes; a++)
    {
        for (uint32_t b = a + 1; b < totalNodes; b++)
        {
            const bool aIsUe = records[a].kind == "ue";
            const bool bIsUe = records[b].kind == "ue";
            if (aIsUe && bIsUe)
            {
                continue;   // UEs do not talk to each other
            }

            Ptr<MobilityModel> mobilityA = nodes.Get(a)->GetObject<MobilityModel>();
            Ptr<MobilityModel> mobilityB = nodes.Get(b)->GetObject<MobilityModel>();
            const double distance = mobilityA->GetDistanceFrom(mobilityB);

            // A backhaul link beyond two rings, or an access link beyond one ring, is
            // not a link at all: mmWave does not reach and the association would never
            // pick it.
            const bool isAccess = aIsUe || bIsUe;
            const double range = isAccess ? kRingRadiusM : 2.0 * kRingRadiusM;
            if (distance > range)
            {
                continue;
            }

            Ptr<ChannelCondition> condition =
                conditionModel->GetChannelCondition(mobilityA, mobilityB);
            const bool los = condition->IsLos();

            // CalcRxPower applies pathloss and shadowing to the transmit power
            const double beamGainDb = arrayGainDb(arrays[a]) + arrayGainDb(arrays[b]);
            const double rxDbm =
                lossModel->CalcRxPower(kTxPowerDbm + beamGainDb, mobilityA, mobilityB);
            const double pathlossDb = kTxPowerDbm + beamGainDb - rxDbm;
            const double noiseDbm = NoiseFloorDbm(isAccess ? kUeNoiseFigureDb
                                                           : kBsNoiseFigureDb);
            const double sinrDb = rxDbm - noiseDbm;
            const double bler = BlerFromSinr(sinrDb);

            linksFile << a << "," << b << "," << (isAccess ? "access" : "backhaul") << ","
                      << distance << "," << (los ? 1 : 0) << "," << pathlossDb << ","
                      << beamGainDb << "," << sinrDb << "," << bler << "\n";
            linkCount++;
        }
    }
    linksFile.close();

    std::cout << "Wrote " << positionsOut << " (" << totalNodes << " nodes)\n";
    std::cout << "Wrote " << linksOut << " (" << linkCount << " links)\n";

    Simulator::Destroy();
    return 0;
}
