import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";

export default function Sidebar() {
    return (
        <aside className="w-64 bg-white p-4 border-r flex flex-col">
            <h2 className="text-lg font-semibold mb-4">Menu</h2>
            <nav className="space-y-2">
                <Button variant="ghost" className="w-full text-left">
                    <Link to="/intro">Intro</Link>
                </Button>
                <Button variant="ghost" className="w-full text-left">
                    <Link to="/scan_rephrase">Scan AI Rephrase</Link>
                </Button>
                <Button variant="ghost" className="w-full text-left">
                    <Link to="/scan_generate">Scan AI Generate</Link>
                </Button>
            </nav>
        </aside>
    );
}
