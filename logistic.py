from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql.functions import col
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# Initialize Spark Session
spark = SparkSession.builder.appName("WineQualityPrediction").getOrCreate()

try:
    # Load data
    data_path = r"C:\Users\TRANS23-LAP-RA1\Downloads\TrainingDataset.csv"
    data = spark.read.csv(data_path, sep=';', header=True, inferSchema=True)
    logging.info("Data loaded successfully.")

    # Clean up column names by removing extra quotes
    for col_name in data.columns:
        data = data.withColumnRenamed(col_name, col_name.replace('"', ''))
    logging.info("Column names cleaned.")

    # Assemble features
    feature_columns = data.columns[:-1]  # all columns except the last one 'quality'
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    feature_data = assembler.transform(data)

    # Scale features
    scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=False)
    scaler_model = scaler.fit(feature_data)
    scaled_data = scaler_model.transform(feature_data)

    # Split data into training and test sets
    (train_data, test_data) = scaled_data.randomSplit([0.8, 0.2], seed=42)
    logging.info("Data split into training and test sets.")

    # Define the model
    lr = LogisticRegression(featuresCol='scaledFeatures', labelCol='quality')

    # Create ParamGrid for Cross Validation
    paramGrid = (ParamGridBuilder()
                 .addGrid(lr.regParam, [0.01, 0.1, 0.5])
                 .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])
                 .addGrid(lr.maxIter, [10, 50, 100])
                 .build())

    # Define evaluator
    evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction", metricName="f1")

    # Set up 3-fold cross-validation
    crossVal = CrossValidator(estimator=lr,
                              estimatorParamMaps=paramGrid,
                              evaluator=evaluator,
                              numFolds=3)

    cvModel = crossVal.fit(train_data)
    logging.info("Model trained.")

    # Use test set here so we can measure the accuracy of our model on new data
    predictions = cvModel.transform(test_data)

    # Evaluate best model
    f1_score = evaluator.evaluate(predictions)
    print("F1 Score: ", f1_score)
    logging.info(f"F1 Score: {f1_score}")

    # Save the best Logistic Regression model
    bestModel = cvModel.bestModel
    bestModel.write().overwrite().save(r"C:\Users\TRANS23-LAP-RA1\model\BestWineQualityModel")
    logging.info("Model saved successfully.")

except Exception as e:
    logging.error("An error occurred:", exc_info=True)

finally:
    # Stop Spark session
    spark.stop()
    logging.info("Spark session stopped.")
