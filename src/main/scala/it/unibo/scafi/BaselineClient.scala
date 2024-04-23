package it.unibo.scafi

import it.unibo.alchemist.model.scafi.ScafiIncarnationForAlchemist._
import it.unibo.scafi.interop.PythonModules.utils
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.{PyQuote, SeqConverters, local}

class BaselineClient
  extends AggregateProgram
  with StandardSensors
  with ScafiAlchemistSupport{

  private lazy val data = utils.get_dataset(indexes())
  private lazy val (trainData, validationData) = splitDataset()
  private val epochs = 2
  private val batch_size = 64

  override def main(): Any = {
    val m = node.get[py.Dynamic]("Model")
    val (evolvedModel, trainLoss) = localTraining(m)
    val (validationAccuracy, validationLoss) = evalModel(evolvedModel)
    logMetrics(trainLoss, validationLoss, validationAccuracy)
    node.put("Model", evolvedModel)
  }

  private def indexes() = node.get[List[Int]](Sensors.data).toPythonProxy

  private def seed(): Int = node.get[Double](Sensors.seed).toInt

  private def logMetrics(trainLoss: Double, validationLoss: Double, validationAccuracy: Double): Unit = {
    node.put(Sensors.trainLoss, trainLoss)
    node.put(Sensors.validationLoss, validationLoss)
    node.put(Sensors.validationAccuracy, validationAccuracy)
  }

  private def localTraining(
     model: py.Dynamic
   ): (py.Dynamic, Double) = {
    val localModel = utils.cnn_loader(seed())
    localModel.load_state_dict(model)
    val result = utils.local_training(localModel, epochs, trainData, batch_size)
    val newWeights = py"$result[0]"
    val trainLoss = py"$result[1]".as[Double]
    (newWeights, trainLoss)
  }

  private def evalModel(model: py.Dynamic): (Double, Double) = {
    val localModel = utils.cnn_loader(seed())
    localModel.load_state_dict(model)
    val result = utils.evaluate(localModel, validationData, batch_size)
    val accuracy = py"$result[0]".as[Double]
    val loss = py"$result[1]".as[Double]
    (accuracy, loss)
  }

  private def splitDataset(): (py.Dynamic, py.Dynamic) = {
    val datasets = utils.train_val_split(data)
    val trainData = py"$datasets[0]"
    val valData = py"$datasets[1]"
    (trainData, valData)
  }

}
