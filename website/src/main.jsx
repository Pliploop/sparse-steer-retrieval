import React from 'react'
import ReactDOM from 'react-dom/client'
import Home from './pages/Home.jsx'
import ErrorBoundary from './components/ErrorBoundary.jsx'
import './index.css'

// Single-page site: the live steerable-retrieval demo is a separate Hugging Face
// Space (linked out), not a bundled route, so no router is needed here.
ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <ErrorBoundary label="Site">
      <Home />
    </ErrorBoundary>
  </React.StrictMode>,
)
