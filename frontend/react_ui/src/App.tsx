import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Sidebar from "@/Sidebar";
import NewScan from "@/NewScan";
import MyScan from '@/Myscan';
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
            <Route path="/" element={<NewScan />} />
            <Route path="/new_scan" element={<NewScan />} />
            <Route path="/my_scans" element={<MyScan />} />
            {/* <Route path="/shared" element={<Shared />} />
            <Route path="/compare" element={<Compare />} /> */}
          </Routes>
        </div>
      </div>
    </Router>
  );
}

export default App;
