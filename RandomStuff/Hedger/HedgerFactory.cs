using MathNet.Numerics.LinearAlgebra;

namespace Hedging
{
    public enum HedgeType
    {
        DV01,
        DV01Convexity,
    }

    public class HedgeFactory
    {
        public static IHedgeMethod GetHedger(HedgeType type)
        {
            return type switch
            {
                HedgeType.DV01 => new DV01Hedge(),
                HedgeType.DV01Convexity => new DV01ConvexityHedge(),
                _ => new DV01Hedge(),
            };
        }
    }
}