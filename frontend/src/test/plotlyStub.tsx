/**
 * A stand-in for the Plotly component factory, used only in tests.
 *
 * The real bundle is megabytes of charting code; parsing and running it inside
 * jsdom made each test file take half a minute and timed the suite out. What a
 * panel test needs to know is that a figure was handed to the chart, which this
 * records in the DOM.
 *
 * Wired up by the `react-plotly.js/factory` alias in vite.config.ts.
 */
interface PlotProps {
  data?: unknown[]
  style?: Record<string, unknown>
}

export default function createPlotlyComponent() {
  return function PlotStub({ data = [] }: PlotProps) {
    return <div data-testid="plot" data-traces={data.length} />
  }
}
