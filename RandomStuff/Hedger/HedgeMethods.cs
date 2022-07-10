using MathNet.Numerics.LinearAlgebra;

namespace Hedging
{
    public interface IHedgeMethod
    {
        public Dictionary<IInstrument, double> ComputeHedgeOrders();
        public void SetBonds(List<IInstrument> instruments);
        public void SetHedgers(List<IInstrument> instruments);
    }

    public abstract class BaseHedgeMethod : IHedgeMethod
    {
        protected  List<IInstrument> _baseInstruments = new();
        protected  List<IInstrument> _hedgeIntruments = new();

        public abstract Dictionary<IInstrument, double> ComputeHedgeOrders();

        public virtual void SetBonds(List<IInstrument> instruments)
        {
            _baseInstruments = instruments;
        }

        public virtual void SetHedgers(List<IInstrument> instruments)
        {
            SetHedgersWithExpectedNumber(instruments, -1);
        }

        protected void SetHedgersWithExpectedNumber(List<IInstrument> instruments, int expectedNumberOfInstruments)
        {
            if (instruments.Count != expectedNumberOfInstruments)
            {
                throw new Exception($"Hedger instruments needs to contain {expectedNumberOfInstruments} bond to use this hedging method.");
            }
            _hedgeIntruments = instruments;
        }

        protected double ComputeBookDV01()
        {
            double val = _baseInstruments.Select(x => x.DV01() * x.Notional).Sum();
            return val;
        }

        protected double ComputeBookConvexity()
        {
            double val = _baseInstruments.Select(x => x.Convexity() * x.Notional).Sum();
            return val;
        }
    }

    public class DV01Hedge : BaseHedgeMethod
    {
        public override Dictionary<IInstrument, double> ComputeHedgeOrders()
        {
            double bookDv01 = ComputeBookDV01();
            double hedgeNotional = - bookDv01 / _hedgeIntruments.First().DV01();

            Dictionary<IInstrument, double> hedgeOrders = new();
            hedgeOrders.Add(_hedgeIntruments.First(), hedgeNotional);
            return hedgeOrders;
        }

        public override void SetBonds(List<IInstrument> instruments)
        {
            base.SetBonds(instruments);
        }

        public override void SetHedgers(List<IInstrument> instruments)
        {
            SetHedgersWithExpectedNumber(instruments, 1);
        }
    }

    public class DV01ConvexityHedge : BaseHedgeMethod
    {
        public override Dictionary<IInstrument, double> ComputeHedgeOrders()
        {
            double bookDv01 = ComputeBookDV01();
            double bookConvexity = ComputeBookConvexity();


            double[,] coefficients = new double[2,2];
            coefficients[0, 0] = _hedgeIntruments[0].DV01();
            coefficients[0, 1] = _hedgeIntruments[1].DV01();
            coefficients[1, 0] = _hedgeIntruments[0].Convexity();
            coefficients[1, 1] = _hedgeIntruments[1].Convexity();

            Matrix<double> A = Matrix<double>.Build.DenseOfArray(coefficients);
            Matrix<double> InverseA = A.Inverse();
            Vector<double> B = Vector<double>.Build.DenseOfArray(new double[] { bookDv01, bookConvexity });
            Vector<double> result = InverseA * B;

            Dictionary<IInstrument, double> hedgeOrders = new()
            {
                { _hedgeIntruments[0], result[0] },
                { _hedgeIntruments[1], result[1] }
            };
            return hedgeOrders;
        }

        public override void SetBonds(List<IInstrument> instruments)
        {
            base.SetBonds(instruments);
        }

        public override void SetHedgers(List<IInstrument> instruments)
        {
            SetHedgersWithExpectedNumber(instruments, 2);
        }
    }
}