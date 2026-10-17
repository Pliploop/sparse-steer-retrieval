import React from 'react'
import ReactDOM from 'react-dom/client'
import Home from './pages/Home.jsx'
import ErrorBoundary from './components/ErrorBoundary.jsx'
import './index.css'

// The project page uses React. Music Lenses is a standalone static page in
// public/lenses/; the live Hugging Face demo remains linked out. No router needed.
ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <ErrorBoundary label="Site">
      <Home />
    </ErrorBoundary>
  </React.StrictMode>,
)
