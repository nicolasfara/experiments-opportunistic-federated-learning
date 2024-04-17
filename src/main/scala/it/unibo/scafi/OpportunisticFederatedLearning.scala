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
  private lazy val data = utils.get_dataset(indexes())
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
  private val batch_size = 256
  private val every = 5
  private val discrepancyThreshold = 2.5 // TODO - check

  override def main(): Any = {
    rep((localModel, 0)) { case (model, tick) =>
      val aggregators = S(
        discrepancyThreshold,
        metric = () => actualMetric(model)
      )
      val (evolvedModel, trainLoss) = localTraining(model)
      node.put("TrainLoss", trainLoss)
      val neighbourhoodMetric = excludingSelf.reifyField(actualMetric(model))
      node.put("NeighbourhoodMetric", neighbourhoodMetric)
      node.put("aggregators", aggregators)
      val potential = classicGradient(aggregators)
      val info = C[Double, Set[py.Dynamic]](
        potential,
        _ ++ _,
        Set(sample(evolvedModel)),
        Set.empty
      )
      val leader = broadcast(aggregators, mid())
      node.put("leader", leader)
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
      model: py.Dynamic
  ): (py.Dynamic, Double) = {
    val result = utils.local_training(model, epochs, data, batch_size)
    val newWeights = py"$result[0]"
    val trainLoss = py"$result[1]".as[Double]
    val freshNN = utils.cnn_loader(seed())
    freshNN.load_state_dict(newWeights)
    (freshNN, trainLoss)
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
    val freshNN = utils.cnn_loader(seed())
    freshNN.load_state_dict(averageWeights)
    freshNN
  }

  private def evalModel(myModel: py.Dynamic): Double = {
    val result = utils.evaluate(myModel, data, batch_size)
    val accuracy = py"$result[0]".as[Double]
    val loss = py"$result[1]".as[Double]
    accuracy
  }

  private def accuracyBasedMetric(model: py.Dynamic): Double = {
    val models = includingSelf.reifyField(nbr(model))
    val evaluations = models.map { case (id, model) => id -> evalModel(model) }
    val neighEvals = excludingSelf.reifyField(nbr(evaluations))
    def directLinkMeToNeigh(): Double =
      neighEvals
        .getOrElse(mid(), Map.empty)
        .getOrElse(nbr(mid()), Double.PositiveInfinity)
    def directLinkNeighToMe(): Double =
      neighEvals
        .getOrElse(nbr(mid()), Map.empty)
        .getOrElse(mid(), Double.PositiveInfinity)
    (directLinkMeToNeigh + directLinkNeighToMe) / 2
  }

  private def snapshot(model: py.Dynamic, id: Int, tick: Int): Unit = {
    torch.save(
      model.state_dict(),
      s"networks/aggregator-$id-time-$tick"
    )
  }

  private def seed(): Int = node.get("seed").toString.toDouble.toInt

  private def indexes() = node.get("data").asInstanceOf[List[Int]].toPythonProxy

  private def impulsesEvery(time: Int): Boolean = time % every == 0

}

object OpportunisticFederatedLearning {
  val DISCREPANCY = "discrepancy"
  val ACCURACY = "accuracy"
}
