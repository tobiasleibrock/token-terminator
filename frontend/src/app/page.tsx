"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Toaster } from "@/components/ui/sonner";
import { toast } from "sonner";
import { Switch } from "@/components/ui/switch";
import { Skeleton } from "@/components/ui/skeleton";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Dialog, DialogContent } from "@/components/ui/dialog";
import { examplePrompts } from "@/lib/utils";
import Image from "next/image";

interface GreenGPTResponse {
  optimizedPrompt: string;
  optimizedAnswer: string;
  originalAnswer: string;
  isCached: boolean;
}

interface AnalyzePromptRequest {
  originalPrompt: string;
  optimizedPrompt: string;
  originalAnswer: string;
  optimizedAnswer: string;
}

interface AnalyzeResponse {
  energySavedWatts: number;
  similarityScoreCosine: number;
  similarityScoreGPT: number;
  originalTokens: number;
  optimizedTokens: number;
}

const API_URL = "https://backend.tokenterminator.deploy.selectcode.dev";

const optimizePrompt = async (prompt: string): Promise<GreenGPTResponse> => {
  const response = await fetch(`${API_URL}/optimize-prompt`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt }),
  });

  if (!response.ok) throw new Error("Failed to optimize prompt");
  return response.json();
};

const analyzePrompt = async (
  data: AnalyzePromptRequest
): Promise<AnalyzeResponse> => {
  const response = await fetch(`${API_URL}/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });

  if (!response.ok) throw new Error("Failed to analyze prompt");
  return response.json();
};

const getEnergySavedVisualization = (energySaved: number) => {
  const scaledEnergySaved = energySaved * 10000;
  const hoursWith6WBulb = scaledEnergySaved / 6; // 6W = 0.006kWh

  if (scaledEnergySaved === 0) {
    return { emoji: "⏳", text: "0 hours saved" };
  } else {
    return {
      emoji: "💡",
      text: `${hoursWith6WBulb.toFixed(1)} hours of a LED bulb`,
    };
  }
};

export default function Home() {
  const [prompt, setPrompt] = useState("");
  const [optimizationResponse, setOptimizationResponse] =
    useState<GreenGPTResponse | null>(null);
  const [analysisResponse, setAnalysisResponse] =
    useState<AnalyzeResponse | null>(null);
  const [showOptimized, setShowOptimized] = useState(true);
  const [showOptimizedAnswer, setShowOptimizedAnswer] = useState(true);
  const [isLoading, setIsLoading] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  // New state variables for accumulative savings
  const [totalTokensSaved, setTotalTokensSaved] = useState(0);
  const [totalEnergySaved, setTotalEnergySaved] = useState(0);

  const [showDemo, setShowDemo] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setAnalysisResponse(null);
    toast("Prompt submitted", {
      description: "Your prompt is being optimized.",
    });

    try {
      const optimizationResult = await optimizePrompt(prompt);
      setOptimizationResponse(optimizationResult);
      setIsLoading(false);

      setIsAnalyzing(true);
      const analysisResult = await analyzePrompt({
        originalPrompt: prompt,
        optimizedPrompt: optimizationResult.optimizedPrompt,
        originalAnswer: optimizationResult.originalAnswer,
        optimizedAnswer: optimizationResult.optimizedAnswer,
      });
      setAnalysisResponse(analysisResult);

      // Update accumulative savings
      const tokensSaved =
        analysisResult.originalTokens - analysisResult.optimizedTokens;
      setTotalTokensSaved((prevTotal) => prevTotal + tokensSaved);
      setTotalEnergySaved(
        (prevTotal) => prevTotal + analysisResult.energySavedWatts
      );
    } catch (error) {
      console.error("Error:", error);
      toast.error("Error", { description: "Failed to process the prompt." });
    } finally {
      setIsLoading(false);
      setIsAnalyzing(false);
    }
  };

  const handleExampleSelect = (selectedPrompt: string) => {
    setPrompt(selectedPrompt);
  };

  const renderPrompt = () => {
    if (!showOptimized || !optimizationResponse) return prompt;

    const originalChars = prompt.split("");
    const optimizedChars = optimizationResponse.optimizedPrompt.split("");
    let optimizedIndex = 0;

    return originalChars.map((char, index) => {
      if (optimizedIndex >= optimizedChars.length) {
        return (
          <span key={index} className="bg-red-300">
            {char}
          </span>
        );
      }

      if (char === optimizedChars[optimizedIndex]) {
        optimizedIndex++;
        return <span key={index}>{char}</span>;
      }

      return (
        <span key={index} className="bg-red-300">
          {char}
        </span>
      );
    });
  };

  if (!showDemo) {
    return (
      <div className="fixed inset-0 bg-gray-100 dark:bg-gray-900 flex flex-col items-center justify-center">
        <h1 className="text-6xl font-bold mb-8 flex items-center">
          <span className="text-7xl mr-4">✂️</span> TokenTerminator
        </h1>
        <Button size="lg" onClick={() => setShowDemo(true)}>
          Go to Demo
        </Button>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex flex-col bg-gray-100 dark:bg-gray-900">
      <header className="bg-white dark:bg-gray-800 shadow">
        <div className="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8">
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
            <span className="text-5xl mr-2">✂️</span> TokenTerminator
          </h1>
        </div>
      </header>

      <main className="flex-grow w-full max-w-[1400px] mx-auto py-6 px-4 sm:px-6 lg:px-8 pb-24">
        <div className="flex flex-col lg:flex-row gap-6">
          <div className="w-full lg:w-2/3">
            <Card className="mb-6">
              <CardHeader>
                <CardTitle className="flex justify-between items-center">
                  <span>📝 Enter your prompt</span>
                  <div className="flex items-center space-x-2">
                    <DropdownMenu>
                      <DropdownMenuTrigger asChild>
                        <Button variant="outline">Example Prompts</Button>
                      </DropdownMenuTrigger>
                      <DropdownMenuContent>
                        {examplePrompts.map((example, index) => (
                          <DropdownMenuItem
                            key={index}
                            onClick={() => handleExampleSelect(example.prompt)}
                          >
                            {example.name}
                          </DropdownMenuItem>
                        ))}
                      </DropdownMenuContent>
                    </DropdownMenu>
                    <Switch
                      id="show-optimized"
                      checked={showOptimized}
                      onCheckedChange={setShowOptimized}
                    />
                    <Label htmlFor="show-optimized">🔍 Show Optimized</Label>
                  </div>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <form onSubmit={handleSubmit}>
                  <div className="grid w-full items-center gap-4">
                    <div className="flex flex-col space-y-1.5">
                      <Label htmlFor="prompt">Prompt</Label>
                      <div className="relative">
                        {!showOptimized || !optimizationResponse ? (
                          <Textarea
                            id="prompt"
                            placeholder="Enter your prompt here or select an example prompt"
                            value={prompt}
                            onChange={(e) => setPrompt(e.target.value)}
                            className="h-48 font-mono text-sm resize-none"
                          />
                        ) : (
                          <div
                            className="h-48 font-mono text-sm rounded-md border border-input bg-transparent px-3 py-2 shadow-sm overflow-auto"
                            style={{
                              lineHeight: "1.5rem",
                              whiteSpace: "pre-wrap",
                              wordBreak: "break-word",
                            }}
                          >
                            {renderPrompt()}
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                  <Button
                    type="submit"
                    className="mt-4"
                    disabled={isLoading || isAnalyzing}
                  >
                    🚀 Submit
                  </Button>
                </form>
              </CardContent>
            </Card>

            {(optimizationResponse || isLoading) && (
              <Card className="mb-6">
                <CardHeader>
                  <CardTitle className="flex justify-between items-center">
                    <div className="flex items-center">
                      <span>🤖 LLM Response</span>
                      {optimizationResponse?.isCached && ( // Changed from 'cached' to 'isCached'
                        <span className="ml-2 px-2 py-1 text-xs font-semibold rounded-full bg-[#147B58] text-white">
                          Cached
                        </span>
                      )}
                    </div>
                    <div className="flex items-center space-x-2">
                      <Switch
                        id="show-optimized-answer"
                        checked={showOptimizedAnswer}
                        onCheckedChange={setShowOptimizedAnswer}
                      />
                      <Label htmlFor="show-optimized-answer">
                        🔍 Show Optimized
                      </Label>
                    </div>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div>
                      <h3 className="font-semibold mb-2">
                        {showOptimizedAnswer
                          ? "Optimized Answer:"
                          : "Original Answer:"}
                      </h3>
                      {isLoading ? (
                        <Skeleton className="h-32 w-full" />
                      ) : (
                        <Textarea
                          value={
                            showOptimizedAnswer
                              ? optimizationResponse?.optimizedAnswer
                              : optimizationResponse?.originalAnswer
                          }
                          readOnly
                          className="h-32 font-mono text-sm resize-none"
                        />
                      )}
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}
          </div>

          <div className="w-full lg:w-1/3">
            <Card>
              <CardHeader>
                <CardTitle>📊 Optimization Statistics</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {/* Per-prompt statistics */}
                  <div>
                    <h3 className="font-semibold">🎟️ Tokens Saved</h3>
                    {isAnalyzing ? (
                      <Skeleton className="h-4 w-20" />
                    ) : analysisResponse ? (
                      <p>
                        {analysisResponse.originalTokens -
                          analysisResponse.optimizedTokens}{" "}
                        (
                        {(
                          ((analysisResponse.originalTokens -
                            analysisResponse.optimizedTokens) /
                            analysisResponse.originalTokens) *
                          100
                        ).toFixed(2)}
                        %)
                      </p>
                    ) : (
                      <p>0 (0%)</p>
                    )}
                  </div>

                  <div>
                    <h3 className="font-semibold">
                      🎯 Similarity Score (Cosine)
                    </h3>
                    {isAnalyzing ? (
                      <Skeleton className="h-4 w-16" />
                    ) : (
                      <p>
                        {(
                          (analysisResponse?.similarityScoreCosine || 0) * 100
                        ).toFixed(2)}
                        %
                      </p>
                    )}
                  </div>
                  <div>
                    <h3 className="font-semibold">🎯 Similarity Score (GPT)</h3>
                    {isAnalyzing ? (
                      <Skeleton className="h-4 w-16" />
                    ) : (
                      <p>
                        {(
                          (analysisResponse?.similarityScoreGPT || 0) * 100
                        ).toFixed(2)}
                        %
                      </p>
                    )}
                  </div>

                  {/* Accumulated and scaled statistics */}
                  <div>
                    <h3 className="font-semibold">🎟️ Tokens Saved (Total)</h3>
                    {isAnalyzing ? (
                      <Skeleton className="h-4 w-20" />
                    ) : (
                      <p>{totalTokensSaved}</p>
                    )}
                  </div>
                  <div>
                    <h3 className="font-semibold">⚡ Energy Saved (Total)</h3>
                    {isAnalyzing ? (
                      <Skeleton className="h-4 w-24" />
                    ) : (
                      <p>{totalEnergySaved.toFixed(4)} Wh</p>
                    )}
                  </div>
                  {/* Horizontal line */}
                  <hr className="border-t border-gray-300 dark:border-gray-700 my-4" />
                  <div>
                    <h3 className="font-semibold">
                      👥 Scaled for 10000 Prompts
                    </h3>
                    {isAnalyzing ? (
                      <>
                        <Skeleton className="h-4 w-32 mb-1" />
                        <Skeleton className="h-4 w-36" />
                      </>
                    ) : (
                      <>
                        <p>Tokens saved: {totalTokensSaved * 10000}</p>
                        <p>
                          Energy: {(totalEnergySaved * 10000).toFixed(2)} Wh
                        </p>
                      </>
                    )}
                  </div>

                  {/* Energy Saved Visualization */}
                  <div className="flex flex-col items-center justify-center mt-4 space-y-2">
                    <span className="text-6xl">
                      {isAnalyzing
                        ? "⏳"
                        : getEnergySavedVisualization(totalEnergySaved).emoji}
                    </span>
                    {isAnalyzing ? (
                      <Skeleton className="h-4 w-32" />
                    ) : (
                      <span className="text-center">
                        {getEnergySavedVisualization(totalEnergySaved).text}
                      </span>
                    )}
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </main>
      <Toaster />
    </div>
  );
}
