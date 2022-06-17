using Microsoft.ML.Data;

namespace Models
{
    public class OnnxModelInput
    {
        [ColumnName("input"), VectorType(12)]
        public float[]? ModelInput { get; set; }
    }

    public class OnnxModelOutput
    {
        [ColumnName("label")]
        public VBuffer<Int64> Label { get; set; }

        [ColumnName("probabilities")]
        public float[]? Probability { get; set; }
    }

}
