namespace UnitTesting
{
    [TestClass]
    public class QlBondWrapperTest
    {
        [TestMethod]
        public void TestYieldBTP()
        {
            double bondRate = 0.9/100.0;
            DateTime maturityDate = new DateTime(2031,4,1);
            DateTime issueDate = new DateTime(2020,10,1);
            var bond = new QlBond(maturityDate, issueDate, bondRate);
            

            double refPrice = 83.11;
            DateTime refDate = new DateTime(2022, 7, 8);
            QlBond.SetEvaluationDate(refDate);

            double expectedYield = 3.15;
            double computedYield = Math.Round(bond.Yield(refPrice) * 100, 2);
            Assert.AreEqual(expectedYield, computedYield);
        }

        public void TestYieldBTP_ShortBond()
        {
            double bondRate = 5.5 / 100.0;
            DateTime issueDate = new DateTime(2012, 3, 1);
            DateTime maturityDate = new DateTime(2022, 9, 1);
            var bond = new QlBond(maturityDate, issueDate, bondRate);

            double refPrice = 100.782;
            DateTime refDate = new DateTime(2022, 7, 8);
            QlBond.SetEvaluationDate(refDate);

            double expectedYield = -0.14;
            double computedYield = Math.Round(bond.Yield(refPrice) * 100, 2);
            Assert.AreEqual(expectedYield, computedYield);
        }

        [TestMethod]
        public void TestYieldBTP_LongBond()
        {
            double bondRate = 3.85 / 100.0;
            DateTime maturityDate = new DateTime(2049, 9, 1);
            DateTime issueDate = new DateTime(2018, 9, 1);
            var bond = new QlBond(maturityDate, issueDate, bondRate);


            double refPrice = 102.79;
            DateTime refDate = new DateTime(2022, 7, 8);
            QlBond.SetEvaluationDate(refDate);

            double expectedYield = 3.72;
            double computedYield = Math.Round(bond.Yield(refPrice) * 100, 2);
            Assert.AreEqual(expectedYield, computedYield);
        }

        [TestMethod]
        public void TestDurationModifiedBTP_ShortBond()
        {
            double bondRate = 9.0 / 100.0;
            DateTime maturityDate = new DateTime(2023, 11, 1);
            DateTime issueDate = new DateTime(1993, 11, 1);
            var bond = new QlBond(maturityDate, issueDate, bondRate);

            double refPrice = 110.356;
            DateTime refDate = new(2022, 7, 8);
            QlBond.SetEvaluationDate(refDate);

            double expectedDuration = 1.24;
            double computedDuration = Math.Round(bond.ModifiedDuration(refPrice), 2);
            Assert.AreEqual(expectedDuration, computedDuration);
        }

        [TestMethod]
        public void TestDurationModifiedBTP_MediumBond()
        {
            double bondRate = 0.9 / 100.0;
            DateTime maturityDate = new DateTime(2031, 4, 1);
            DateTime issueDate = new DateTime(2020, 10, 1);
            var bond = new QlBond(maturityDate, issueDate, bondRate);

            double refPrice = 83.11;
            DateTime refDate = new DateTime(2022, 7, 8);
            QlBond.SetEvaluationDate(refDate);

            double computedDurationModified = Math.Round(bond.ModifiedDuration(refPrice), 2);
            Assert.AreEqual(8.09, computedDurationModified);
        }

        [TestMethod]
        public void TestDurationModifiedBTP_LongBond()
        {
            double bondRate = 3.85 / 100.0;
            DateTime maturityDate = new DateTime(2049, 9, 1);
            DateTime issueDate = new DateTime(2018, 9, 1);
            var bond = new QlBond(maturityDate, issueDate, bondRate);

            double refPrice = 102.79;
            DateTime refDate = new DateTime(2022, 7, 8);
            QlBond.SetEvaluationDate(refDate);

            double expectedDuration = 16.39;
            double computedDuration = Math.Round(bond.ModifiedDuration(refPrice), 2);
            Assert.AreEqual(expectedDuration, computedDuration);
        }

        [TestMethod]
        public void TestDurationSimpleBTP()
        {
            double bondRate = 0.9 / 100.0;
            DateTime maturityDate = new DateTime(2031, 4, 1);
            DateTime issueDate = new DateTime(2020, 10, 1);
            var bond = new QlBond(maturityDate, issueDate, bondRate);

            double refPrice = 83.11;
            DateTime refDate = new DateTime(2022, 7, 8);
            QlBond.SetEvaluationDate(refDate);

            double computedDurationSimple = Math.Round(bond.SimpleDuration(refPrice), 2);
            Assert.AreEqual(8.35, computedDurationSimple);
        }

        [TestMethod]
        public void TestDurationMacaulayBTP()
        {
            double bondRate = 0.9 / 100.0;
            DateTime maturityDate = new DateTime(2031, 4, 1);
            DateTime issueDate = new DateTime(2020, 10, 1);
            var bond = new QlBond(maturityDate, issueDate, bondRate);

            double refPrice = 83.11;
            DateTime refDate = new DateTime(2022, 7, 8);
            QlBond.SetEvaluationDate(refDate);

            double computedDurationMacaulay = Math.Round(bond.MacaulayDuration(refPrice), 2);
            Assert.AreEqual(8.35, computedDurationMacaulay);
        }

        [TestMethod]
        public void TestPriceBTP()
        {
            double bondRate = 0.9 / 100.0;
            DateTime maturityDate = new DateTime(2031, 4, 1);
            DateTime issueDate = new DateTime(2020, 10, 1);
            var bond = new QlBond(maturityDate, issueDate, bondRate);

            DateTime refDate = new DateTime(2022, 7, 8);
            QlBond.SetEvaluationDate(refDate);

            double rate = 3.15 / 100.0;
            double computedPrice = Math.Round(bond.CleanPrice(rate), 3);

            Assert.AreEqual(83.11, computedPrice, 0.01);
        }

        [TestMethod]
        public void TestBasisPointValueBTP()
        {
            double bondRate = 0.9 / 100.0;
            DateTime maturityDate = new(2031, 4, 1);
            DateTime issueDate = new(2020, 10, 1);
            var bond = new QlBond(maturityDate, issueDate, bondRate);

            double refPrice = 83.11;
            DateTime refDate = new DateTime(2022, 7, 8);
            QlBond.SetEvaluationDate(refDate);

            double expectedBasisPointValue = 6.74;
            double basisPointValue = - 100 * bond.BasisPointValue(refPrice);
            Assert.AreEqual(expectedBasisPointValue, basisPointValue, 0.01);
        }

        [TestMethod]
        public void TestDV01BTP()
        {
            double bondRate = 0.9 / 100.0;
            DateTime maturityDate = new(2031, 4, 1);
            DateTime issueDate = new(2020, 10, 1);
            var bond = new QlBond(maturityDate, issueDate, bondRate);

            double refPrice = 83.11;
            DateTime refDate = new(2022, 7, 8);
            QlBond.SetEvaluationDate(refDate);

            double actualDV01 = 6.75;
            double dv01 = bond.DV01(refPrice);
            Assert.AreEqual(dv01, actualDV01, 0.01);
        }

        [TestMethod]
        public void TestConvexityBTP()
        {
            double bondRate = 0.9 / 100.0;
            DateTime maturityDate = new(2031, 4, 1);
            DateTime issueDate = new(2020, 10, 1);
            var bond = new QlBond(maturityDate, issueDate, bondRate);

            double refPrice = 83.11;
            DateTime refDate = new(2022, 7, 8);
            QlBond.SetEvaluationDate(refDate);

            double actualConvexity = 75.34;
            double convexity = bond.Convexity(refPrice);
            Assert.AreEqual(convexity, actualConvexity, 0.01);
        }

        [TestMethod]
        public void TestYearsToMaturity()
        {
            double bondRate = 0.9 / 100.0;
            DateTime maturityDate = new(2031, 4, 1);
            DateTime issueDate = new(2020, 10, 1);
            var bond = new QlBond(maturityDate, issueDate, bondRate);
            DateTime refDate = new(2022, 7, 8);
            QlBond.SetEvaluationDate(refDate);

            double ytm = bond.YearsToMaturity();
            Assert.AreEqual(ytm, 8.73, 0.01);
        }

    }
}