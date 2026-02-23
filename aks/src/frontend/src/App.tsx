import { Dashboard } from './components/Dashboard'

const globalStyles = `
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { overflow: hidden; }
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
  }
  ::-webkit-scrollbar { width: 8px; }
  ::-webkit-scrollbar-track { background: #0d1117; }
  ::-webkit-scrollbar-thumb { background: #1e2a3a; border-radius: 4px; }
  ::-webkit-scrollbar-thumb:hover { background: #2e3a4a; }
`;

function App() {
  return (
    <>
      <style>{globalStyles}</style>
      <Dashboard />
    </>
  )
}

export default App
