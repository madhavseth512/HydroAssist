/**
 * A figure the server drew.
 *
 * The API sends Plotly JSON built by the registry's own plot adapters, so the
 * curve on screen is the same one the PDF prints — there is no second
 * implementation of any model in the browser.
 *
 * The chart library itself is loaded on first use (see PlotlyChart.tsx).
 */
import { Alert, Box, Skeleton } from '@mantine/core'
import { Suspense, lazy, useMemo } from 'react'

const Plot = lazy(() => import('./PlotlyChart'))

interface Figure {
  data: unknown[]
  layout: Record<string, unknown>
}

export interface PlotFigureProps {
  /** Plotly JSON, or null while it is being computed. */
  figure?: string | null
  height?: number
  loading?: boolean
  /** Shown when there is no figure — a fit without a picture is still a fit. */
  emptyMessage?: string
}

export function PlotFigure({
  figure,
  height = 460,
  loading = false,
  emptyMessage = 'No figure to show yet.',
}: PlotFigureProps) {
  const parsed = useMemo<Figure | null>(() => {
    if (!figure) return null
    try {
      return JSON.parse(figure) as Figure
    } catch {
      return null
    }
  }, [figure])

  if (loading) return <Skeleton height={height} radius="sm" />

  if (!parsed) {
    return (
      <Alert color="gray" variant="light">
        {emptyMessage}
      </Alert>
    )
  }

  return (
    <Box>
      <Suspense fallback={<Skeleton height={height} radius="sm" />}>
        <Plot
          data={parsed.data as Plotly.Data[]}
          layout={{
            ...(parsed.layout as object),
            autosize: true,
            // The server sets the figure's own height; the container decides
            // the width so it reflows on a narrow screen.
            margin: (parsed.layout as { margin?: object } | undefined)?.margin,
          }}
          style={{ width: '100%', height }}
          useResizeHandler
          config={{ displaylogo: false, responsive: true }}
        />
      </Suspense>
    </Box>
  )
}
