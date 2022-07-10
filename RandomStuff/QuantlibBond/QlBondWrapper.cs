using QLNet;


namespace QuantlibBond
{
    public class QlBond
    {
        private readonly FixedRateBond _fixedRateBond;
        private readonly DayCounter _accrualDayCounter;
        private readonly Frequency _evaluationFrequency;
        private readonly Compounding _evaluationCompounding;

        public QlBond(DateTime maturityDate, DateTime issueDate, double fixedRate)
        {
            int settlementDays = 2;
            double faceAmount = 100;

            Date issue = new(issueDate);
            Date maturity = new(maturityDate);
            Period tenor = new(Frequency.Semiannual);
            
            Calendar calendar = new NullCalendar();
            BusinessDayConvention convention = BusinessDayConvention.Unadjusted;
            BusinessDayConvention maturityDateConvention = BusinessDayConvention.Unadjusted;
            BusinessDayConvention paymentConvention = BusinessDayConvention.ModifiedFollowing;
            DateGeneration.Rule rule = DateGeneration.Rule.Backward;

            Schedule schedule = new(issue, maturity, tenor, calendar, convention, maturityDateConvention, rule, false);
            _accrualDayCounter = new ActualActual(ActualActual.Convention.Bond, schedule);
            List<double> coupons = new() { fixedRate };

            _fixedRateBond = new FixedRateBond(settlementDays, faceAmount, schedule, coupons, _accrualDayCounter, paymentConvention);
            _evaluationFrequency = Frequency.Annual;
            _evaluationCompounding = Compounding.Compounded;
        }

        public static void SetEvaluationDate(Date date)
        {
            QLNet.Settings.setEvaluationDate(date);
        }

        public double Yield(double price)
        {
            double yield = _fixedRateBond.yield(price, _accrualDayCounter, _evaluationCompounding, _evaluationFrequency);
            return yield;
        }

        public double ModifiedDuration(double price)
        {
            double rate = Yield(price);
            InterestRate yield = new(rate, _accrualDayCounter, _evaluationCompounding, _evaluationFrequency);
            double duration = BondFunctions.duration(_fixedRateBond, yield, Duration.Type.Modified);
            return duration;
        }

        public double SimpleDuration(double price)
        {
            double rate = Yield(price);
            InterestRate yield = new(rate, _accrualDayCounter, _evaluationCompounding, _evaluationFrequency);
            double duration = BondFunctions.duration(_fixedRateBond, yield, Duration.Type.Simple);
            return duration;
        }

        public double MacaulayDuration(double price)
        {
            double rate = Yield(price);
            InterestRate yield = new(rate, _accrualDayCounter, _evaluationCompounding, _evaluationFrequency);
            double duration = BondFunctions.duration(_fixedRateBond, yield, Duration.Type.Macaulay);
            return duration;
        }

        public double DV01(double price)
        {
            return -100 * BasisPointValue(price);
        }

        public double BasisPointValue(double price)
        {
            double rate = Yield(price);
            InterestRate yield = new(rate, _accrualDayCounter, _evaluationCompounding, _evaluationFrequency);
            double bpv = BondFunctions.basisPointValue(_fixedRateBond, yield);
            return bpv;
        }

        public double Convexity(double price)
        {
            double r = Yield(price);
            InterestRate yield = new(r, _accrualDayCounter, _evaluationCompounding, _evaluationFrequency);
            double convexity = BondFunctions.convexity(_fixedRateBond, yield);
            return convexity;
        }

        public double CleanPrice(double rate)
        {
            InterestRate yield = new(rate, _accrualDayCounter, _evaluationCompounding, _evaluationFrequency);
            double price = BondFunctions.cleanPrice(_fixedRateBond, yield);
            return price;
        }

        public double YearsToMaturity()
        {
            double ytm = _accrualDayCounter.yearFraction(Settings.evaluationDate(), _fixedRateBond.maturityDate());
            return ytm;
        }
    }
}