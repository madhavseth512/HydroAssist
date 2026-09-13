/**
 * The Plotly component, in a module of its own so it can be loaded lazily.
 *
 * Plotly is most of the application's JavaScript. Keeping it out of the first
 * bundle means the workspace paints before any chart library has downloaded,
 * which matters on the free-tier host every visitor fetches it from.
 */
import Plotly from 'plotly.js-basic-dist-min'
import createPlotlyComponent from 'react-plotly.js/factory'

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const Plot = createPlotlyComponent(Plotly as any)

export default Plot
