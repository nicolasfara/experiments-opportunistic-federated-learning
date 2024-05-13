package it.unibo.scafi

import it.unibo.alchemist.model.scafi.ScafiIncarnationForAlchemist._
import interop.PythonModules._
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.{PyQuote, SeqConverters}
import Sensors._
import it.unibo.alchemist.model.layers.Dataset
import it.unibo.alchemist.model.scafi.ScafiIncarnationForAlchemist

class OpportunisticFederatedLearning
    extends AggregateProgram
    with StandardSensors
    with ScafiAlchemistSupport
    with FieldUtils
    with TimeUtils
    with BuildingBlocks {

  private lazy val localModel = {
    val result = utils.cnn_loader(seed)
    println(result.state_dict())
    result
  }
  private def data = senseEnvData[Dataset](Sensors.phenomena)
  def trainData = data.trainingData
  def validationData = data.validationData
  private val boundedInt: Builtins.Bounded[ID] = Builtins.Bounded.of_i
  private val boundedDouble: Builtins.Bounded[Double] = Builtins.Bounded.of_d
  implicit val boundedOfTuple: ScafiIncarnationForAlchemist.Builtins.Bounded[
    (Double, Int, Double, Double)
  ] = Builtins.Bounded.tupleBounded4(
    boundedDouble,
    boundedInt,
    boundedDouble,
    boundedDouble
  )
  private def actualMetric: (py.Dynamic) => () => Double = (model) => {
    val models = includingSelf.reifyField(nbr(model))
    val evaluations = models.map { case (id, model) =>
      id -> evalModel(model, trainData)._2
    }
    val neighEvals = includingSelf.reifyField(nbr(evaluations))
    () => accuracyBasedMetric(neighEvals)
  }

  private lazy val epochs = sense[Int](Sensors.epochs)
  private lazy val batch_size = sense[Int](Sensors.batchSize)
  private lazy val aggregateLocalEvery = sense[Int](Sensors.aggregateLocalEvery)
  private lazy val threshold = sense[Double](lossThreshold)

  override def main(): Any = {
    rep((localModel, localModel, 1)) { case (local, global, tick) =>
      val metric = actualMetric(local)
      val leader = SWithMinimisingShare(
        threshold,
        metric = metric,
        symBreaker = mid()
      )
      val isAggregator = leader == mid()
      val (evolvedModel, trainLoss) = localTraining(local, trainData)
      val (validationAccuracy, validationLoss) =
        evalModel(evolvedModel, validationData)
      val neighbourhoodMetric = excludingSelf.reifyField(metric())
      val potential = classicGradient(isAggregator, metric)
      // flexGradient(0.5, 0.9, 1)(isAggregator, metric)
      val sender = G_along(potential, metric, mid(), (_: ID) => nbr(mid()))
      val info = CWithSenderField[List[py.Dynamic]](
        _ ++ _,
        List(evolvedModel),
        List.empty,
        sender
      )
      val areaId = data.areaId
      val aggregatedModel = averageWeights(info, List.fill(info.length)(1.0))
      val sharedModel = broadcast(isAggregator, aggregatedModel, metric)

      if (isAggregator) { snapshot(sharedModel, mid(), tick) }
      // Actuations
      val (same, _) = rep((false, leader)) { case (same, oldId) =>
        (oldId == leader) -> leader
      }
      node.put(Sensors.sameLeader, same)
      node.put(Sensors.leaderId, leader)
      node.put(Sensors.model, sample(local))
      node.put(Sensors.sharedModel, sample(sharedModel))
      node.put("Federation", leader % 9)
      node.put(Sensors.areaId, areaId)
      if (isAggregator) { node.put(models, info) }
      node.put(Sensors.potential, potential)
      node.put(Sensors.modelsCount, info.length)
      node.put(Sensors.neighbourhoodMetric, neighbourhoodMetric)
      node.put(Sensors.isAggregator, isAggregator)
      node.put(Sensors.trainLoss, trainLoss)
      node.put(Sensors.validationLoss, validationLoss)
      node.put(Sensors.validationAccuracy, validationAccuracy)
      node.put(Sensors.parent, findParentWithMetric(potential, metric))
      node.put(Sensors.sender, sender)
      mux(impulsesEvery(tick)) {
        (
          averageWeights(List(evolvedModel, sharedModel), List(0.1, 0.9)),
          averageWeights(List(evolvedModel, sharedModel), List(0.1, 0.9)),
          tick + 1
        )
      } {
        (
          evolvedModel,
          averageWeights(List(evolvedModel, sharedModel), List(0.9, 0.1)),
          tick + 1
        )
      }
    }
  }

  private def localTraining(
      model: py.Dynamic,
      trainData: py.Dynamic
  ): (py.Dynamic, Double) = {
    val result =
      utils.local_training(model, epochs, trainData, batch_size, seed)
    val newWeights = py"$result[0]"
    val trainLoss = py"$result[1]".as[Double]
    val freshNN = utils.cnn_loader(seed)
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

  private def averageWeights(
      models: List[py.Dynamic],
      weights: List[Double]
  ): py.Dynamic = {
    val averageWeights =
      utils.average_weights(
        models.map(sample).toPythonProxy,
        weights.toPythonProxy
      )
    val freshNN = utils.cnn_loader(seed)
    freshNN.load_state_dict(averageWeights)
    freshNN
  }

  private def evalModel(
      myModel: py.Dynamic,
      validationData: py.Dynamic
  ): (Double, Double) = {
    val result = utils.evaluate(myModel, validationData, batch_size, seed)
    val accuracy = py"$result[0]".as[Double]
    val loss = py"$result[1]".as[Double]
    (accuracy, loss)
  }

  private def accuracyBasedMetric(
      neighEvals: Map[ID, Map[ID, Double]]
  ): Double = {
    def directLinkMeToNeigh(): Double =
      neighEvals
        .getOrElse(mid(), Map.empty)
        .getOrElse(nbr(mid()), Double.PositiveInfinity)
    def directLinkNeighToMe(): Double =
      neighEvals
        .getOrElse(nbr(mid()), Map.empty)
        .getOrElse(mid(), Double.PositiveInfinity)
    (directLinkMeToNeigh + directLinkNeighToMe)
  }

  private def snapshot(model: py.Dynamic, id: Int, tick: Int): Unit = {
    torch.save(
      model.state_dict(),
      s"networks/aggregator-$id-time-$tick"
    )
  }

  private lazy val seed: Int = node.get[Double](Sensors.seed).toInt

  private def impulsesEvery(time: Int): Boolean =
    time % aggregateLocalEvery == 0

  def CWithMetric[V](
      potential: Double,
      acc: (V, V) => V,
      local: V,
      Null: V,
      metric: () => Double
  ): V =
    share(local) { case (_, query) =>
      acc(
        local,
        foldhood(Null)(acc) {
          mux(nbr(findParentWithMetric(potential, metric)) == mid()) {
            query()
          } { nbr(Null) }
        }
      )
    }

  def CWithSenderField[V](
      acc: (V, V) => V,
      local: V,
      Null: V,
      parentField: ID
  ): V = share(local) { case (_, query) =>
    acc(
      local,
      foldhoodPlus(Null)(acc) {
        mux(nbr(parentField) == mid()) {
          query()
        } { nbr(Null) }
      }
    )
  }

  def findParentWithMetric(potential: Double, metric: () => Double): ID = {
    val others =
      excludingSelf.reifyField((potential - nbr(potential)).similarTo(metric()))
    mux(potential.similarTo(0)) {
      Int.MaxValue
    } {
      others.find(_._2).map(_._1).getOrElse(Int.MaxValue)
    }
  }

  implicit class RichDouble(d: Double) {
    def similarTo(other: Double, precision: Double = 0.01): Boolean =
      d < (other + precision) && other < (d + precision)
  }

  import BoundedMessage._
  import BoundedMessage.BoundedMsg
  import Builtins.Bounded

  // cf. https://arxiv.org/pdf/1711.08297.pdf
  def SWithMinimisingShare(grain: Double, symBreaker: Int, metric: Metric): ID = {
    def fMP(value: Candidacy): Candidacy = value match {
      case Candidacy(_, dd, id) if id == mid() || dd >= grain => BoundedMessage.BoundedMsg.top
      case m                                                  => m
    }

    val loc = Candidacy(symBreaker, 0.0, mid())
    share[Candidacy](loc) { case (_, nbrc) =>
      minHoodPlusLoc(loc) {
        val nbrCandidacy = nbrc()
        fMP(nbrCandidacy.copy(distance = nbrCandidacy.distance + metric()))
      }
    }.leaderId
  }
}
import Builtins.Bounded
private case class Candidacy(symBreaker: Int, distance: Double, leaderId: Int)

private object BoundedMessage {
  implicit object BoundedMsg extends Bounded[Candidacy] {
    override def bottom: Candidacy =
      Candidacy(implicitly[Bounded[Int]].bottom, implicitly[Bounded[Double]].bottom, implicitly[Bounded[ID]].bottom)

    override def top: Candidacy =
      Candidacy(implicitly[Bounded[Int]].top, implicitly[Bounded[Double]].top, implicitly[Bounded[ID]].top)

    override def compare(a: Candidacy, b: Candidacy): Int =
      List(a.symBreaker.compareTo(b.symBreaker), a.distance.compareTo(b.distance), a.leaderId.compareTo(b.leaderId))
        .collectFirst { case x if x != 0 => x }
        .getOrElse(0)
  }
}

object ColorRandomUtils {
  // write 10 very different colors
  val paletteOfDifferentColorsHue = List(
    0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f
  )
  def colorFromPalette(index: Int): Float =
    paletteOfDifferentColorsHue(index % paletteOfDifferentColorsHue.size)
}
