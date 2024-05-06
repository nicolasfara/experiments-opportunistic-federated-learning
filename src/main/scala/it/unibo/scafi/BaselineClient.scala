package it.unibo.scafi

import it.unibo.alchemist.model.layers.Dataset
import it.unibo.alchemist.model.scafi.ScafiIncarnationForAlchemist._
import it.unibo.scafi.interop.PythonModules.utils
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.{PyQuote, SeqConverters, local}

class BaselineClient
    extends AggregateProgram
    with StandardSensors
    with ScafiAlchemistSupport {

  private def data = senseEnvData[Dataset](Sensors.phenomena)
  def trainData = data.trainingData
  def validationData = data.validationData
  private lazy val epochs = sense[Int](Sensors.epochs)
  private lazy val batch_size = sense[Int](Sensors.batchSize)

  override def main(): Any = {
    val m = node.get[py.Dynamic](Sensors.model)
    val (evolvedModel, trainLoss) = localTraining(m)
    val (validationAccuracy, validationLoss) = evalModel(evolvedModel)
    logMetrics(trainLoss, validationLoss, validationAccuracy)
    node.put(Sensors.model, evolvedModel)
  }

  private def seed(): Int = node.get[Double](Sensors.seed).toInt

  private def logMetrics(
      trainLoss: Double,
      validationLoss: Double,
      validationAccuracy: Double
  ): Unit = {
    node.put(Sensors.trainLoss, trainLoss)
    node.put(Sensors.validationLoss, validationLoss)
    node.put(Sensors.validationAccuracy, validationAccuracy)
  }

  private def localTraining(
      model: py.Dynamic
  ): (py.Dynamic, Double) = {
    val localModel = utils.cnn_loader(seed())
    localModel.load_state_dict(model)
    val result =
      utils.local_training(localModel, epochs, trainData, batch_size, seed())
    val newWeights = py"$result[0]"
    val trainLoss = py"$result[1]".as[Double]
    (newWeights, trainLoss)
  }

  private def evalModel(model: py.Dynamic): (Double, Double) = {
    val localModel = utils.cnn_loader(seed())
    localModel.load_state_dict(model)
    val result = utils.evaluate(localModel, validationData, batch_size, seed())
    val accuracy = py"$result[0]".as[Double]
    val loss = py"$result[1]".as[Double]
    (accuracy, loss)
  }

}
