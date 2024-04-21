import argparse
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def main(args):
    spark = SparkSession.builder.appName("WineQualityPrediction").getOrCreate()

    # Load the model
    model = PipelineModel.load(args.model_path)

    # Load validation data
    test_data = spark.read.csv(args.test_data_path, header=True, inferSchema=True)

    # Predict
    predictions = model.transform(test_data)

    # Evaluate
    evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction", metricName="f1")
    f1_score = evaluator.evaluate(predictions)
    print("F1 Score:", f1_score)

    spark.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Wine Quality Prediction Model Testing')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--test_data_path', type=str, required=True, help='Path to the test dataset')
    
    args = parser.parse_args()
    main(args)
