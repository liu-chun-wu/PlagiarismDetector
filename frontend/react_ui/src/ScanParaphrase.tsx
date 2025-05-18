import { useState } from "react";
import { useRef } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Textarea } from "@/components/ui/textarea";
import { Upload } from "lucide-react";

let upload_text_API = import.meta.env.VITE_API_URL_TEXT_PARAPHRASED;
let upload_pdf_API = import.meta.env.VITE_API_URL_PDF_PARAPHRASED;
upload_text_API = upload_text_API.replace(/\\x3a/g, ":");
upload_pdf_API = upload_pdf_API.replace(/\\x3a/g, ":");

// Skeleton component
const Skeleton = ({ className }: { className?: string }) => {
    return (
        <div className={`animate-pulse bg-gray-300 rounded ${className}`} />
    );
};

export default function ScanParaphrase() {
    const [uploaded, setUploaded] = useState<boolean>(false);
    const [aiContent, setAiContent] = useState<number>(0);
    const [confidenceScore, setConfidenceScore] = useState<number>(0);
    const [textInput, setTextInput] = useState<string>("");
    const [highlightedText, setHighlightedText] = useState<string>("");
    const [loading, setLoading] = useState<boolean>(false);
    const [pdfFile, setPdfFile] = useState<File | null>(null);
    const fileInputRef = useRef<HTMLInputElement | null>(null);
    const [verdict, setVerdict] = useState<{ result: string; reason: string } | null>(null);


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

            if (!response.ok) throw new Error("Failed to upload text");

            const data = await response.json();
            setAiContent(data.plagiarism_percentage || 0);
            setConfidenceScore(data.plagiarism_confidence || 0);
            setUploaded(true);
            setVerdict(data.verdict || null);

            // ✅ 使用 unified 函數處理
            highlightSnippetsUnified(data.original_text_and_plagiarism_snippet, data.verdict?.result);
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

            if (!response.ok) throw new Error("Failed to upload PDF");

            const data = await response.json();

            if (data.original_text_and_plagiarism_snippet?.[0]?.original_text) {
                setTextInput(data.original_text_and_plagiarism_snippet[0].original_text);
            }

            setAiContent(data.plagiarism_percentage || 0);
            setConfidenceScore(data.plagiarism_confidence || 0);
            setUploaded(true);
            setVerdict(data.verdict || null);

            // ✅ 使用 unified 函數處理
            highlightSnippetsUnified(data.original_text_and_plagiarism_snippet, data.verdict?.result);
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

    const highlightSnippetsUnified = (
        originalTextAndSnippets: { original_text: string; plagiarism_snippet: string[] }[],
        verdictResult: string | null = null
    ): void => {
        const results: string[] = [];

        originalTextAndSnippets.forEach(({ original_text, plagiarism_snippet }) => {
            // 若判定為 SOURCE，整段加上 highlight
            if (verdictResult === "SOURCE") {
                const full = `<span class="bg-yellow-300">${original_text}</span>`;
                results.push(full);
                return;
            }

            // 若沒有 snippet，原樣顯示
            if (!plagiarism_snippet || plagiarism_snippet.length === 0) {
                results.push(original_text);
                return;
            }

            let matches: [number, number][] = [];

            plagiarism_snippet.forEach((snippet) => {
                const windowSize = snippet.length;
                const threshold = 0.7;

                for (let i = 0; i <= original_text.length - windowSize; i++) {
                    const substring = original_text.slice(i, i + windowSize);
                    const similarity = jaccardSimilarity(substring.toLowerCase(), snippet.toLowerCase());

                    if (similarity >= threshold) {
                        matches.push([i, i + windowSize - 1]);
                    }
                }
            });

            // 合併重疊範圍
            matches.sort((a, b) => a[0] - b[0]);
            const merged: [number, number][] = [];
            for (const [start, end] of matches) {
                if (merged.length === 0 || start > merged[merged.length - 1][1]) {
                    merged.push([start, end]);
                } else {
                    merged[merged.length - 1][1] = Math.max(merged[merged.length - 1][1], end);
                }
            }

            // 插入 highlight HTML
            let highlighted = "";
            let currentIndex = 0;
            for (const [start, end] of merged) {
                highlighted += original_text.slice(currentIndex, start);
                highlighted += `<span class="bg-yellow-300">${original_text.slice(start, end + 1)}</span>`;
                currentIndex = end + 1;
            }
            highlighted += original_text.slice(currentIndex);
            results.push(highlighted);
        });

        setHighlightedText(results.join("<br/><br/>"));
    };



    const handleNewScan = () => {
        setUploaded(false);
        setAiContent(0);
        setConfidenceScore(0);
        setTextInput("");
        setHighlightedText("");
    };

    return (
        <div className="flex min-h-screen bg-gray-100">
            <main className="flex-1 p-8">
                <h1 className="text-4xl font-bold text-center text-blue-700 mb-6">Detect AI Paraphrased Content</h1>
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
                        <CardContent className="mt-2">
                            <div className="mt-4">
                                <p className="font-semibold">Original Text with Highlighted Plagiarism Snippet</p>
                                <div
                                    className="p-4 bg-gray-200 rounded-md mt-2 max-h-[400px] max-w-[700px] overflow-y-auto overflow-x-auto whitespace-pre-wrap text-base leading-relaxed"
                                    dangerouslySetInnerHTML={{ __html: highlightedText }}
                                />
                                <p className="mt-4 font-semibold">Plagiarism Percentage</p>
                                <Progress value={aiContent} className="mt-2 bg-red-500" />
                                <p className="text-right font-bold text-red-600">{aiContent}%</p>
                                <p className="mt-4 font-semibold">Confidence Score</p>
                                <Progress value={confidenceScore} className="mt-2 bg-blue-500" />
                                <p className="text-right font-bold text-blue-600">{confidenceScore}%</p>
                                {verdict && (
                                    <>
                                        {/* Verdict Block */}
                                        <p className="text-base font-semibold text-gray-700">Verdict</p>
                                        <p
                                            className={`text-xl font-bold mt-2 ${verdict.result === "ACCEPT"
                                                ? "text-green-600"
                                                : verdict.result === "ABSTAIN"
                                                    ? "text-red-600"
                                                    : verdict.result === "SOURCE"
                                                        ? "text-blue-600"
                                                        : "text-gray-600"
                                                }`}
                                        >
                                            {verdict.result}
                                        </p>

                                        {/* Spacer */}
                                        <div className="h-4" />

                                        {/* Explanation Block */}
                                        <p className="text-base font-semibold text-gray-700">Explanation</p>
                                        <p className="mt-2 text-gray-800 leading-relaxed">{verdict.reason}</p>
                                    </>
                                )}
                                <Button className="mt-6" onClick={handleNewScan}>New Scan</Button>
                            </div>
                        </CardContent>
                    </Card>
                )}
            </main>
        </div>
    );
}
