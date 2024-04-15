package it.unibo.scafi

import me.shadaj.scalapy.py

object PythonModules {
  val utils = py.module("FLutils")
  val torch = py.module("torch")
}
