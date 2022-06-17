using Microsoft.ML;

namespace Models
{
    public class OnnxModel
    {
        private readonly MLContext _mlContext;
        private readonly ITransformer _mlModel;

        public OnnxModel(string modelFilePath)
        {
            _mlContext = new MLContext();
            _mlModel = SetupModelTransformer(modelFilePath);
        }

        private ITransformer SetupModelTransformer(string modelPath)
        {
            var pipeline = _mlContext
                .Transforms
                .ApplyOnnxModel(modelPath);
            ITransformer mlModel = pipeline.Fit(CreateEmptyDataView());
            return mlModel;
        }

        private IDataView CreateEmptyDataView()
        {
            // Used to initialized a pretrained model.
            List<OnnxModelInput> list = new() { new OnnxModelInput() { ModelInput = Array.Empty<float>() } };
            return _mlContext.Data.LoadFromEnumerable(list);
        }

        public IEnumerable<OnnxModelOutput> Predict(List<OnnxModelInput> input)
        {
            IDataView dataView = _mlContext.Data.LoadFromEnumerable(input);
            return Predict(dataView);
        }

        public IEnumerable<OnnxModelOutput> Predict(IDataView input)
        {
            var transformedValues = _mlModel.Transform(input);
            var predictions = _mlContext.Data.CreateEnumerable<OnnxModelOutput>(transformedValues, reuseRowObject: true);
            return predictions;
        }
    }
}


