import { useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Upload } from "lucide-react";
import { Textarea } from "@/components/ui/textarea";

const uploadAPI = import.meta.env.VITE_API_URL_UPLOAD;

export default function NewScan() {
    const [uploaded, setUploaded] = useState<boolean>(false);
    const [aiContent, setAiContent] = useState<number>(100);
    const [confidenceScore, setConfidenceScore] = useState<number>(100);
    const [plagiarismSnippet, setPlagiarismSnippet] = useState<string>("");
    const [textInput, setTextInput] = useState<string>("");
    const [highlightedText, setHighlightedText] = useState<string>("");

    const handleUpload = async () => {
        try {
            const response = await fetch(uploadAPI, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ text: textInput }),
            });

            if (!response.ok) {
                throw new Error("Failed to upload text");
            }

            const data = await response.json();
            setAiContent(data.plagiarism_percentage || 100);
            setConfidenceScore(data.confidence_score || 100);
            setPlagiarismSnippet(data.plagiarism_snippet || "");
            setUploaded(true);
            highlightPlagiarism(data.plagiarism_snippet);
        } catch (error) {
            console.error("Error uploading text:", error);
            alert("Failed to upload text");
        }
    };

    const highlightPlagiarism = (plagiarismSnippet: string) => {
        if (!plagiarismSnippet) {
            setHighlightedText(textInput);
            return;
        }

        const regex = new RegExp(`(${plagiarismSnippet})`, "gi");
        const highlighted = textInput.replace(regex, '<span class="bg-yellow-300">$1</span>');
        setHighlightedText(highlighted);
    };

    const handleNewScan = () => {
        setUploaded(false);
        setAiContent(100);
        setConfidenceScore(100);
        setPlagiarismSnippet("");
        setTextInput("");
        setHighlightedText("");
    };

    return (
        <div className="flex min-h-screen bg-gray-100">
            {/* Main Content */}
            <main className="flex-1 p-8">
                {!uploaded ? (
                    <div className="flex flex-col items-center justify-center h-full">
                        <Upload className="w-20 h-20 text-blue-600" />
                        <p className="text-xl mt-4">PASTE YOUR TEXT HERE</p>
                        <div className="flex space-x-4 mt-4 w-full max-w-md">
                            <Textarea
                                className="w-full p-2 border rounded-lg bg-white"
                                placeholder="Paste your text here..."
                                value={textInput}
                                onChange={(e) => setTextInput(e.target.value)}
                            />
                        </div>
                        <Button className="mt-6" onClick={handleUpload}>Upload</Button>
                    </div>
                ) : (
                    <Card className="p-6 max-w-3xl mx-auto">
                        <h2 className="text-2xl font-bold">Plagiarism Detection Results</h2>
                        <CardContent className="mt-4">
                            <div className="mt-6">
                                <p className="font-semibold">Original Text with Highlighted Plagiarism Snippet</p>
                                <div className="p-2 bg-gray-200 rounded-md mt-2" dangerouslySetInnerHTML={{ __html: highlightedText }} />
                                <p className="mt-4 font-semibold">Plagiarism Percentage</p>
                                <Progress value={aiContent} className="mt-2 bg-red-500" />
                                <p className="text-right font-bold text-red-600">{aiContent}%</p>
                                <p className="mt-4 font-semibold">Confidence Score</p>
                                <Progress value={confidenceScore} className="mt-2 bg-blue-500" />
                                <p className="text-right font-bold text-blue-600">{confidenceScore}%</p>
                                <Button className="mt-6" onClick={handleNewScan}>New Scan</Button>
                            </div>
                        </CardContent>
                    </Card>
                )}
            </main>
        </div>
    );
}