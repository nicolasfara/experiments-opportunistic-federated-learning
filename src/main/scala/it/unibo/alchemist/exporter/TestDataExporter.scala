package it.unibo.alchemist.exporter

import com.github.tototoshi.csv.CSVWriter
import java.io.File

object TestDataExporter {

  def CSVExport(data: List[Double], path: String): Unit = {
    val f = new File(s"$path.csv")
    val writer = CSVWriter.open(f)
    writer.writeAll(List(data))
    writer.close()
  }

}
