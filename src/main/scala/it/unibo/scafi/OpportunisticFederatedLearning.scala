package it.unibo.scafi

import it.unibo.alchemist.model.scafi.ScafiIncarnationForAlchemist._
import interop.PythonModules._
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.{PyQuote, SeqConverters}

class OpportunisticFederatedLearning
    extends AggregateProgram
    with StandardSensors
    with ScafiAlchemistSupport
    with FieldUtils
    with BuildingBlocks {

  private lazy val localModel = utils.cnn_loader(seed())
  private lazy val metricSelection: String =
    node.get[String]("metric")
  private lazy val actualMetric: (py.Dynamic) => Double =
    metricSelection match {
      case OpportunisticFederatedLearning.DISCREPANCY =>
        (model) => discrepancyMetric(model, nbr(model))
      case OpportunisticFederatedLearning.ACCURACY =>
        (model) => accuracyBasedMetric(model)
    }
  private val epochs = 2
  private val discrepancyThreshold = 1.3 // TODO - check

  override def main(): Any = {
    val data = utils.get_dataset(indexes())
    rep((localModel, 0)) { case (model, tick) =>
      val aggregators = S(
        discrepancyThreshold,
        metric = () => actualMetric(model)
      )
      val (evolvedModel, trainLoss, valLoss) = localTraining(model, data)
      node.put("TrainLoss", trainLoss)
      node.put("ValidationLoss", valLoss)
      val potential = classicGradient(aggregators)
      val info = C[Double, Set[py.Dynamic]](
        potential,
        _ ++ _,
        Set(sample(evolvedModel)),
        Set.empty
      )
      val aggregatedModel = averageWeights(info)
      val sharedModel = broadcast(aggregators, aggregatedModel)
      if (aggregators) { snapshot(sharedModel, mid(), tick) }
      mux(impulsesEvery(tick)) {
        (
          averageWeights(Set(sample(sharedModel), sample(evolvedModel))),
          tick + 1
        )
      } {
        (evolvedModel, tick + 1)
      }
    }
  }

  private def localTraining(
      model: py.Dynamic,
      data: py.Dynamic
  ): (py.Dynamic, Double, Double) = {
    val result = utils.local_train(model, epochs, data) // TODO - implement py
    val trainLoss = py"$result[0]".as[Double]
    val valLoss = py"$result[1]".as[Double]
    val newWeights = py"$result[2]"
    val freshNN = utils.cnn_factory()
    freshNN.load_state_dict(newWeights)
    (freshNN, trainLoss, valLoss)
  }

  private def discrepancyMetric(
      myModel: py.Dynamic,
      otherModule: py.Dynamic
  ): Double = {
    val discrepancy =
      utils.discrepancy(myModel.state_dict(), otherModule.state_dict())
    py"$discrepancy".as[Double]
  }

  private def sample(model: py.Dynamic): py.Dynamic =
    model.state_dict()

  private def averageWeights(models: Set[py.Dynamic]): py.Dynamic = {
    val averageWeights =
      utils.average_weights(models.toSeq.toPythonProxy)
    val freshNN = utils.cnn_factory()
    freshNN.load_state_dict(averageWeights)
    freshNN
  }

  private def evalModel(myModel: py.Dynamic): Double = ??? // TODO - implement

  private def accuracyBasedMetric(model: py.Dynamic): Double = {
    val models = includingSelf.reifyField(nbr(model))
    val evaluations = models.map { case (id, model) => id -> evalModel(model) }
    val neighEvals = excludingSelf.reifyField(nbr(evaluations))
    (neighEvals(mid())(nbr(mid())) + neighEvals(nbr(mid()))(mid())) / 2
  }

  private def snapshot(model: py.Dynamic, id: Int, tick: Int): Unit = {
    torch.save(
      model.state_dict(),
      s"networks/aggregator-$id-$tick"
    )
  }

  private def seed(): Int = node.get("seed").toString.toDouble.toInt

  private def indexes() = node.get("data").asInstanceOf[List[Int]].toPythonProxy
}

object OpportunisticFederatedLearning {
  private val DISCREPANCY = "discrepancy"
  private val ACCURACY = "accuracy"
}
