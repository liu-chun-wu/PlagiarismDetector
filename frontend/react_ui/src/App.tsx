import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Sidebar from "@/Sidebar";
import ScanGenerate from "@/ScanGenerate";
import ScanRephrase from "@/ScanRephrase";
import Intro from '@/Intro';
// import MyScans from "./pages/MyScans";
// import Shared from "./pages/Shared";
// import Compare from "./pages/Compare";

function App() {
  return (
    <Router>
      <div className="flex min-h-screen">
        {/* Sidebar 永遠存在 */}
        <Sidebar />

        {/* 主要內容區域 */}
        <div className="flex-1">
          <Routes>
            <Route path="/" element={<Intro />} />
            <Route path="/intro" element={<Intro />} />
            <Route path="/scan_rephrase" element={<ScanRephrase />} />
            <Route path="/scan_generate" element={<ScanGenerate />} />
          </Routes>
        </div>
      </div>
    </Router>
  );
}

export default App;
