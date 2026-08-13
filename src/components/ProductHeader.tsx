export function ProductHeader() {
  return (
    <header className="product-global-header">
      <div className="global-product-identity">
        <div className="global-product-name">
          <span>AGENTIC KNOWLEDGE</span>
          <strong>多模态PDF检索</strong>
        </div>
      </div>

      <div className="global-product-actions">
        <div className="technology-signature" aria-label="项目核心技术">
          <span>POWERED BY</span>
          <div>
            <strong className="tech-deepseek"><i />DeepSeek</strong>
            <em>×</em>
            <strong className="tech-langchain"><i />LangChain</strong>
            <em>×</em>
            <strong className="tech-paddle"><i />PaddleOCR</strong>
          </div>
        </div>
      </div>
    </header>
  );
}
