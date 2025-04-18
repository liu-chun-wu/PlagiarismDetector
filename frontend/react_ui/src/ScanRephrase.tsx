import { useState } from "react";
import { useRef } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Textarea } from "@/components/ui/textarea";
import { Upload } from "lucide-react";

const upload_text_API = import.meta.env.VITE_API_URL_TEXT_REPHRASED;
const upload_pdf_API = import.meta.env.VITE_API_URL_PDF_REPHRASED;

// Skeleton component
const Skeleton = ({ className }: { className?: string }) => {
    return (
        <div className={`animate-pulse bg-gray-300 rounded ${className}`} />
    );
};

export default function ScanRephrase() {
    const [uploaded, setUploaded] = useState<boolean>(false);
    const [aiContent, setAiContent] = useState<number>(100);
    const [confidenceScore, setConfidenceScore] = useState<number>(100);
    const [textInput, setTextInput] = useState<string>("");
    const [highlightedText, setHighlightedText] = useState<string>("");
    const [loading, setLoading] = useState<boolean>(false);
    const [pdfFile, setPdfFile] = useState<File | null>(null);
    const fileInputRef = useRef<HTMLInputElement | null>(null);


    const handleTextUpload = async () => {
        setLoading(true);
        try {
            const response = await fetch(upload_text_API, {
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
            setConfidenceScore(data.avg_confidence || 100);
            setUploaded(true);
            highlightPlagiarism(data.plagiarism_snippet);
        } catch (error) {
            console.error("Error uploading text:", error);
            alert("Failed to upload text");
        } finally {
            setLoading(false);
        }
    };

    const handlePDFUpload = async () => {
        if (!pdfFile) {
            alert("Please select a PDF file first.");
            return;
        }

        const formData = new FormData();
        formData.append("file", pdfFile);

        setLoading(true);
        try {
            const response = await fetch(upload_pdf_API, {
                method: "POST",
                body: formData,
            });

            if (!response.ok) {
                throw new Error("Failed to upload PDF");
            }

            const data = await response.json();
            setAiContent(data.plagiarism_percentage || 100);
            setConfidenceScore(data.avg_confidence || 100);
            setUploaded(true);
            highlightPlagiarism(data.plagiarism_snippet);
        } catch (error) {
            console.error("Error uploading PDF:", error);
            alert("Failed to upload PDF");
        } finally {
            setLoading(false);
        }
    };


    const jaccardSimilarity = (str1: string, str2: string): number => {
        const set1 = new Set(str1);
        const set2 = new Set(str2);
        const intersection = new Set([...set1].filter(x => set2.has(x)));
        const union = new Set([...set1, ...set2]);
        return intersection.size / union.size;
    };

    const highlightPlagiarism = (plagiarismSnippet: string): void => {
        if (!plagiarismSnippet) {
            setHighlightedText(textInput);
            return;
        }

        const windowSize = plagiarismSnippet.length;
        const threshold = 0.7;
        const matches: [number, number][] = [];

        for (let i = 0; i <= textInput.length - windowSize; i++) {
            const substring = textInput.slice(i, i + windowSize);
            const similarity = jaccardSimilarity(substring.toLowerCase(), plagiarismSnippet.toLowerCase());

            if (similarity >= threshold) {
                matches.push([i, i + windowSize - 1]);
            }
        }

        if (matches.length === 0) {
            setHighlightedText(textInput);
            return;
        }

        let highlightedText = "";
        let currentIndex = 0;

        matches.sort((a, b) => a[0] - b[0]).forEach(([start, end]) => {
            if (start < currentIndex) {
                return;
            }
            highlightedText += textInput.slice(currentIndex, start);
            highlightedText += `<span class="bg-yellow-300">${textInput.slice(start, end + 1)}</span>`;
            currentIndex = end + 1;
        });

        highlightedText += textInput.slice(currentIndex);
        setHighlightedText(highlightedText);
    };

    const handleNewScan = () => {
        setUploaded(false);
        setAiContent(100);
        setConfidenceScore(100);
        setTextInput("");
        setHighlightedText("");
    };

    return (
        <div className="flex min-h-screen bg-gray-100">
            <main className="flex-1 p-8">
                <h1 className="text-4xl font-bold text-center text-blue-700 mb-6">Detect AI Rephrased Content</h1>
                {loading ? (
                    <Card className="p-6 max-w-3xl mx-auto">
                        <CardContent className="mt-4 space-y-6">
                            <Skeleton className="h-8 w-3/4" />
                            <Skeleton className="h-32 w-full" />
                            <div>
                                <Skeleton className="h-4 w-1/2 mb-2" />
                                <Skeleton className="h-4 w-full" />
                            </div>
                            <div>
                                <Skeleton className="h-4 w-1/2 mb-2" />
                                <Skeleton className="h-4 w-full" />
                            </div>
                            <Skeleton className="h-10 w-32" />
                        </CardContent>
                    </Card>
                ) : !uploaded ? (
                    <div className="flex flex-1 flex-col items-center justify-center">
                        {/* Upload 圖示 */}
                        <Upload className="w-20 h-20 text-blue-600 mb-6" />

                        {/* 上傳區塊（兩個） */}
                        <div className="flex flex-row space-x-8 w-full max-w-4xl justify-center">
                            {/* PDF 區塊 */}
                            <div className="flex flex-col justify-between border border-gray-300 rounded-xl p-4 shadow-md bg-white w-full max-w-sm h-[300px]">
                                <div className="flex flex-col items-center space-y-2">
                                    <p className="text-lg font-semibold text-blue-800">Upload PDF File</p>
                                    <Button
                                        onClick={() => fileInputRef.current?.click()}
                                        variant="default"
                                    >
                                        Select PDF File
                                    </Button>
                                    <input
                                        type="file"
                                        accept="application/pdf"
                                        ref={fileInputRef}
                                        onChange={(e) => {
                                            if (e.target.files?.[0]) setPdfFile(e.target.files[0]);
                                        }}
                                        className="hidden"
                                    />
                                    {pdfFile && <span className="text-xl text-blue-700 font-medium">{pdfFile.name}</span>}
                                </div>
                                <Button onClick={handlePDFUpload}>Upload PDF</Button>
                            </div>


                            {/* Text 區塊 */}
                            <div className="flex flex-col justify-between border border-gray-300 rounded-xl p-4 shadow-md bg-white w-full max-w-sm h-[300px]">
                                <div className="flex flex-col items-center space-y-2 w-full">
                                    <p className="text-lg font-semibold text-blue-800">Upload Text</p>
                                    <Textarea
                                        className="w-full p-2 border rounded-lg bg-white h-40 resize-none overflow-y-auto"
                                        placeholder="Paste your text here..."
                                        value={textInput}
                                        onChange={(e) => setTextInput(e.target.value)}
                                    />
                                </div>
                                <Button onClick={handleTextUpload}>Upload Text</Button>
                            </div>
                        </div>
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
