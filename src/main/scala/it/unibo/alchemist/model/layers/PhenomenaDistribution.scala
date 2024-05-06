package it.unibo.alchemist.model.layers

import it.unibo.alchemist.model._
import it.unibo.scafi.interop.PythonModules
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.{PyQuote, SeqConverters}

class PhenomenaDistribution[P <: Position[P]](
    environment: Environment[_, P],
    private val xStart: Double,
    private val yStart: Double,
    private val xEnd: Double,
    private val yEnd: Double,
    val areas: Int,
    private val samplesPerArea: Int,
    private val shuffle: Boolean = true,
    private val dataFraction: Double = 1.0,
    private val seed: Int,
    private val train: Boolean
) extends Layer[Dataset, P] {

  private lazy val phenomenaAreas: List[(P, P)] =
    computeSubAreas(
      environment.makePosition(xStart, yStart),
      environment.makePosition(xEnd, yEnd),
      areas
    )

  /** The key is the ID of the phenomena area, matching the [samplesPerArea] size
    */
  private lazy val splitLabelsAndIndices: Map[Int, List[(Int, Int)]] = {
    PythonModules.utils
      .dataset_to_nodes_partitioning(areas, seed, shuffle, train, dataFraction)
      .as[Map[Int, List[(Int, Int)]]]
  }

  lazy val dataByPositions: Map[P, Dataset] = {
    phenomenaAreas.zipWithIndex.flatMap { case ((startPoint, endPoint), rawIndex) =>
      def areaIndex: Int = if (rawIndex >= areas) areas - 1 else rawIndex
      val positionsWithinArea =
        computeSubAreas(startPoint, endPoint, samplesPerArea).zipWithIndex
      val totalDataPerPoint =
        splitLabelsAndIndices(areaIndex).size / samplesPerArea
      val dataPerArea =
        splitLabelsAndIndices(areaIndex).grouped(totalDataPerPoint).toList
      positionsWithinArea.map { case (position, index) =>
        val data = dataPerArea(index).map(_._1)
        val split = PythonModules.utils.train_val_split(
          PythonModules.utils.get_dataset(data.toPythonProxy, train),
          seed
        )
        center(position) -> Dataset(
          areaIndex,
          py"${split}[0]",
          py"${split}[1]"
        )
      }
    }.toMap
  }

  def cleanAll(): Unit =
    dataByPositions.values.foreach { data =>
      data.trainingData.del()
      data.validationData.del()
    }
  private def computeSubAreas(start: P, end: P, areas: Int): List[(P, P)] = {
    val rows = math.sqrt(areas).toInt
    val cols = (areas + rows - 1) / rows
    val width = math.abs(end.getCoordinate(0) - start.getCoordinate(0))
    val height = math.abs(end.getCoordinate(1) - start.getCoordinate(1))
    val rowHeight = height / rows
    val colWidth = width / cols

    val result = for {
      row <- 0 until rows
      col <- 0 until cols
    } yield {
      val x1 = start.getCoordinate(0) + col * colWidth
      val y1 = start.getCoordinate(1) + row * rowHeight
      val x2 = x1 + colWidth
      val y2 = y1 + rowHeight
      environment.makePosition(x1, y1) -> environment.makePosition(x2, y2)
    }
    result.toList
  }

  private def center(p: (P, P)): P = {
    val xCenter = (p._1.getCoordinate(0) + p._2.getCoordinate(0)) / 2
    val yCenter = (p._1.getCoordinate(1) + p._2.getCoordinate(1)) / 2
    environment.makePosition(xCenter, yCenter)
  }

  override def getValue(p: P): Dataset = dataByPositions
    .map { case (position, dataset) => position.distanceTo(p) -> dataset }
    .minBy { case (distance, _) => distance }
    ._2
}

case class Dataset(
    areaId: Int,
    trainingData: py.Dynamic,
    validationData: py.Dynamic
)
