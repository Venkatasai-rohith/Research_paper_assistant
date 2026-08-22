/**
 * Normalizes mathematical formulas and LaTeX delimiters in markdown text
 * so that remark-math, rehype-katex, and remark-gfm can parse them properly.
 */
export function preprocessMarkdown(content: string): string {
  if (!content) return "";

  let text = content;

  // 1. Normalize explicit LaTeX display math \[ ... \] to $$ ... $$
  text = text.replace(/\\\[([\s\S]*?)\\\]/g, (_, equation) => {
    return `\n\n$$\n${equation.trim()}\n$$\n\n`;
  });

  // 2. Normalize explicit LaTeX inline math \( ... \) to $ ... $
  text = text.replace(/\\\(([\s\S]*?)\\\)/g, (_, equation) => {
    return ` $${equation.trim()}$ `;
  });

  // 3. Catch raw bracket-wrapped equations like:
  // [ \text{PE}(pos, 2i) = \sin!\left(\frac{pos}{...}\right) ... ]
  text = text.replace(
    /(?:^|\n)\s*\[\s*(\\(?:text|frac|mathbf|mathcal|mathrm|sum|int|sin|cos|tan|log|exp|sqrt|alpha|beta|gamma|theta|sigma|omega|Delta|nabla|partial|approx|sim|le|ge|ne|in|times|pm|cdot)[\s\S]*?|\b[A-Za-z_][A-Za-z0-9_]*\s*=\s*\\[\s\S]*?)\]\s*(?:\n|$)/g,
    (_, equation) => {
      return `\n\n$$\n${equation.trim()}\n$$\n\n`;
    }
  );

  // 4. Ensure display math $$ ... $$ has empty lines before and after for clean AST parsing
  text = text.replace(/([^\n])\n*\$\$\n*([\s\S]*?)\n*\$\$/g, '$1\n\n$$\n$2\n$$');
  text = text.replace(/\$\$\n*([\s\S]*?)\n*\$\$([^\n])/g, '$$\n$1\n$$\n\n$2');

  // 5. Ensure tables have a blank line before and after so remark-gfm always triggers
  text = text.replace(/([^\n])\n(\|[^\n]+\|\n\|[-:\s|]+\|)/g, '$1\n\n$2');

  return text;
}
