using MathNet.Numerics.LinearAlgebra;
using QuantlibBond;


namespace Hedging
{
    public interface IInstrument
    {
        public double Notional { get; set; }
        public double DollarDuration();
        public double DV01();
        public double Convexity();
        public double YearsToMaturity();
    }

    public class HFuture : IInstrument
    {
        private readonly IInstrument _cheapestToDeliver;
        private readonly double _conversionFactor;

        public HFuture(IInstrument cheapestToDeliver, double conversionFactor)
        {
            _cheapestToDeliver = cheapestToDeliver;
            _conversionFactor = conversionFactor;
        }

        public double Notional { get; set; }

        public double Price { get; set; }

        public double Convexity()
        {
            return _cheapestToDeliver.Convexity() / _conversionFactor;
        }

        public double DollarDuration()
        {
            return _cheapestToDeliver.DollarDuration() / _conversionFactor;
        }

        public double DV01()
        {
            return _cheapestToDeliver.DV01() / _conversionFactor;
        }

        public double YearsToMaturity()
        {
            return _cheapestToDeliver.YearsToMaturity();
        }
    }

    public class HQlBondWrapper : IInstrument
    {
        private readonly QlBond _bond;

        public HQlBondWrapper(QlBond bond)
        {
            _bond = bond;
        }

        public double Notional { get; set; }

        public double Price { get; set; }

        public double Convexity()
        {
            return _bond.Convexity(Price);
        }

        public double DollarDuration()
        {
            double duration = _bond.SimpleDuration(Price);
            double dollarDuration = duration / (Price / 100);
            return dollarDuration;
        }

        public double DV01()
        {
            return _bond.DV01(Price);
        }

        public double YearsToMaturity()
        {
            return _bond.YearsToMaturity();
        }
    }
}