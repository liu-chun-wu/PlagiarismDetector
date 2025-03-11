import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";

export default function Sidebar() {
    return (
        <aside className="w-64 bg-white p-4 border-r flex flex-col">
            <h2 className="text-lg font-semibold mb-4">Menu</h2>
            <nav className="space-y-2">
                <Button variant="ghost" className="w-full text-left">
                    <Link to="/new_scan">New Scan</Link>
                </Button>
                <Button variant="ghost" className="w-full text-left">
                    <Link to="/my_scans">My Scans</Link>
                </Button>
                <Button variant="ghost" className="w-full text-left">
                    <Link to="/">Shared</Link>
                </Button>
                <Button variant="ghost" className="w-full text-left">
                    <Link to="/">Compare</Link>
                </Button>
            </nav>
        </aside>
    );
}
