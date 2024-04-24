package it.unibo.alchemist.model.layers

import it.unibo.alchemist.model._
import it.unibo.scafi.interop.PythonModules

class PhenomenaDistribution[P <: Position[P]](
    environment: Environment[_, P],
    private val xStart: Double,
    private val yStart: Double,
    private val xEnd: Double,
    private val yEnd: Double,
    private val areas: Int,
    private val samplesPerArea: Int,
    private val shuffle: Boolean = true,
    private val dataFraction: Double = 1.0,
    private val seed: Int
) extends Layer[(Int, List[Int]), P] {

  private lazy val phenomenaAreas: List[(P, P)] =
    computeSubAreas(environment.makePosition(xStart, yStart), environment.makePosition(xEnd, yEnd), areas)

  /** The key is the ID of the phenomena area, matching the [samplesPerArea] size
    */
  private lazy val splitLabelsAndIndices: Map[Int, List[(Int, Int)]] = {
    PythonModules.utils
      .dataset_to_nodes_partitioning(areas, seed, shuffle, dataFraction)
      .as[Map[Int, List[(Int, Int)]]]
  }

  private lazy val dataByPositions: Map[(P, Int), List[(Int, Int)]] = {
    phenomenaAreas.zipWithIndex.flatMap { case ((startPoint, endPoint), areaIndex) =>
      val positionsWithinArea = computeSubAreas(startPoint, endPoint, samplesPerArea).zipWithIndex
      val dataPerArea = splitLabelsAndIndices(areaIndex).grouped(samplesPerArea).toList
      positionsWithinArea.map { case (position, index) =>
        val data = dataPerArea(index)
        (center(position), areaIndex) -> data
      }
    }.toMap
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

  override def getValue(p: P): (Int, List[Int]) = dataByPositions
    .map { case ((position, areaIndex), values) => (position.distanceTo(p), areaIndex) -> values.map(_._1) }
    .minBy { case ((distance, _), _) => distance } match { case ((_, id), ids) => id -> ids }
}
