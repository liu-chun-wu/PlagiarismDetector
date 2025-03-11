import { Card, CardContent } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { useState } from "react";

const PlagiarismResult = () => {
    const [plagiarismPercentage, setPlagiarismPercentage] = useState(80); // 假設 80% 抄襲
    const [aiGeneratedPercentage, setAiGeneratedPercentage] = useState(100); // AI 內容 100%

    const textWithHighlights = [
        {
            text: "Research adopts questionnaire survey method ",
            highlight: true,
        },
        {
            text: "to base on advertising attitude & rational behavior theory ",
            highlight: false,
        },
        {
            text: "establish questionnaire investigates Taiwan users’ acceptance ",
            highlight: true,
        },
        {
            text: "of social media e-commerce platform with using intention.",
            highlight: false,
        },
    ];

    return (
        <div className="p-6 max-w-3xl mx-auto">
            <Card>
                <CardContent className="p-6">
                    <h2 className="text-2xl font-bold mb-4">Plagiarism Detection Results</h2>

                    {/* 抄襲內容高亮顯示 */}
                    <div className="border p-4 rounded-lg bg-gray-50">
                        {textWithHighlights.map((chunk, index) => (
                            <span
                                key={index}
                                className={chunk.highlight ? "bg-yellow-200 px-1" : ""}
                            >
                                {chunk.text}
                            </span>
                        ))}
                    </div>

                    {/* 抄襲百分比顯示 */}
                    <div className="mt-4">
                        <p className="font-semibold">Plagiarism Detected: {plagiarismPercentage}%</p>
                        <Progress value={plagiarismPercentage} className="w-full" />
                    </div>

                    {/* AI 內容判斷 */}
                    <div className="mt-4">
                        <p className="font-semibold">AI Content Found: {aiGeneratedPercentage}%</p>
                        <Progress value={aiGeneratedPercentage} className="w-full bg-blue-200" />
                    </div>
                </CardContent>
            </Card>
        </div>
    );
};

export default PlagiarismResult;
