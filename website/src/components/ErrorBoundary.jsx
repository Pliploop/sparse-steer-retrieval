import { Component } from 'react'

/**
 * Scope a boundary around anything risky so one throwing component (or a failed
 * lazy chunk) shows a named red box instead of blanking the whole page.
 */
export default class ErrorBoundary extends Component {
  constructor(props) {
    super(props)
    this.state = { error: null }
  }

  static getDerivedStateFromError(error) {
    return { error }
  }

  componentDidCatch(error, info) {
    console.error('[site] component failed:', error, info?.componentStack)
  }

  render() {
    if (!this.state.error) return this.props.children
    return (
      <div className="rounded-2xl border border-dashed border-red-300 bg-red-50/60 p-6 text-sm dark:border-red-900/60 dark:bg-red-950/20">
        <p className="font-semibold text-red-700 dark:text-red-400">
          {this.props.label ?? 'This section'} failed to render.
        </p>
        <p className="mt-1 text-xs text-red-600/80 dark:text-red-400/70">{String(this.state.error)}</p>
      </div>
    )
  }
}
