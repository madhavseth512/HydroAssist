/// <reference types="vite/client" />

/**
 * The basic Plotly bundle ships no types of its own.
 *
 * It is the basic build deliberately: the full plotly.js is several megabytes,
 * and every figure HydroAssist draws is scatter, line or bar on linear or log
 * axes. The free-tier host serves this file to every visitor.
 */
declare module 'plotly.js-basic-dist-min' {
  const plotly: unknown
  export default plotly
}
