export default function Intro() {
    return (
        <div className="p-8 max-w-4xl mx-auto text-gray-800">
            {/* Logo 和標題 */}
            <div className="text-center mb-10">
                <img
                    src="/lab_logo.jpg"
                    alt="Lab Logo"
                    className="w-80 h-auto mx-auto mb-4"
                />
                <h1 className="text-3xl font-serif font-bold mb-2">中文學術論文 AI 生成與改寫辨識工具</h1>
                <p className="text-center text-sm text-gray-500 mt-2">
                    國立中央大學, 人工智慧與知識系統實驗室
                </p>
                <p className="text-center text-sm text-gray-500 mt-2">
                    National Central University, Artificial Intelligence and Knowledge System Lab
                </p>
            </div>

            {/* 系統簡介 */}
            <section className="mb-8">
                <h2 className="text-xl font-semibold mb-2 border-b pb-1">系統簡介</h2>
                <ul className="list-disc list-inside space-y-1">
                    <li>本系統包含兩大功能： 偵測論文內容 1.是否有被 AI 改寫 和 2.是否由 AI 生成</li>
                    <li>系統基於 Multi-agent 架構，並將其部署為一個易用的網頁服務，讓用戶能夠快速上傳文本並獲得偵測結果</li>
                </ul>
            </section>

            {/* 使用方法 & 可檢測範圍 */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
                <section>
                    <h2 className="text-xl font-semibold mb-2 border-b pb-1">使用方法</h2>
                    <ol className="list-decimal list-inside space-y-1">
                        <li>上傳欲檢測之檔案(檔案格式為文本或是PDF)</li>
                        <li>查看分析的結果</li>
                    </ol>
                </section>
                <section>
                    <h2 className="text-xl font-semibold mb-2 border-b pb-1">使用資料</h2>
                    <ul className="list-disc list-inside space-y-1">
                        <li>台灣博碩士論文知識加值系統的論文</li>
                        <li>學校: 中央大學(NCU)、中山大學(NSYU)、中正大學(CCU)、政治大學(NCCU)、陽明交通大學(NYCU)</li>
                        <li>學門: 電算機學門、商業及管理學門</li>
                    </ul>
                </section>
            </div>
            <section className="mb-8">
                <h2 className="text-xl font-semibold mb-2 border-b pb-1">指導教授</h2>
                <p> 楊鎮華 老師​</p>
            </section>
            <section className="mb-8">
                <h2 className="text-xl font-semibold mb-2 border-b pb-1">開發團隊</h2>
                <p>王廷安、劉俊吾、張祐嘉、張耘碩​</p>
            </section>
        </div>
    );
}
