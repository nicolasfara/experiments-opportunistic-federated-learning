package it.unibo.scafi.interop

import me.shadaj.scalapy.py

object PythonModules {
  val utils: py.Module = py.module("FLutils")
  val torch = py.module("torch")
}
