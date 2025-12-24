# Tutorial: Creating HUST Beamer Slides in LaTeX

## 1. Document Setup and Configuration

The "Preamble" (the top part of your code) controls the global settings of your presentation.

### Basic Class and Packages

Start your document by defining the class and necessary packages.

```latex
% 't' aligns text to top, 'aspectratio=169' sets widescreen (16:9)
\documentclass[t, aspectratio=169]{beamer}

% Essential for typing Vietnamese
\usepackage[utf8]{vietnam}

% Math packages for equations and symbols
\usepackage{amsthm,amsmath,amssymb}
\usepackage{lmodern} % scalable font

```

### Loading the HUST Theme

This is where you load the custom design. You can change the color and aspect ratio here.

```latex
% Options:
% Color: [blue], [red], [green] (depending on what the .sty file supports)
% Ratio: [169] for widescreen, [43] for standard 4:3
\usepackage[blue, 169]{beamerthemeHUST}

```

### Margin Configuration

The template requires specific margin definitions to align text with the custom background/header.

```latex
\def\slideContentLeftMargin{0.5cm}   % Left margin
\def\slideContentRightMargin{1cm}    % Right margin
\def\hustHeaderHeight{1cm}         % Top margin (avoids header)
\def\slideContentBotMargin{3.5cm}    % Bottom margin (avoids logo)

```

---

## 2. Setting Up the Title Page

You need to input your information (Title, Author, Institute) and configure how they align on the cover slide.

### Input Data

```latex
\title{YOUR PROJECT TITLE HERE}
% \subtitle{Optional Subtitle}
\author{Student Name 1 - ID \\ Student Name 2 - ID}
\institute{Your Department / Class / Club}
\date{\today}

```

### Alignment Configuration

You can shift the text block or change alignment (Left, Center, Right).

```latex
% Shift the entire text block coordinates
\renewcommand{\titlePageTitleX}{-0.5cm}
\renewcommand{\titlePageTitleY}{-0.5cm}

% Align specific elements (options: \alignLeft, \alignCenter, \alignRight)
\renewcommand{\hustTitleAlign}{\alignLeft}
\renewcommand{\hustSubtitleAlign}{\alignLeft}
\renewcommand{\hustAuthorAlign}{\alignLeft}
\renewcommand{\hustInstAlign}{\alignLeft}

```

---

## 3. Structure of the Presentation

Everything visible goes inside `\begin{document} ... \end{document}`.

### Opening Slides

The template provides specific commands for the intro sequence:

1. `\hustintropage{}`: A splash screen/intro slide.
2. `\hustsectionpage{}`: Often used to signal a new section, but placed at the start here.
3. `\husttitlepage`: Generates the actual Title Slide with the data you entered in Step 2.
4. **Table of Contents:**
```latex
\begin{frame}[allowframebreaks]{Outline}
    \tableofcontents
\end{frame}

```



---

## 4. Creating Content Slides

A standard slide is created using the `frame` environment.

### Basic Text Slide with Lists

```latex
\section{Section Name} % Appears in TOC and header

\begin{frame}{Slide Title}
    \textbf{Bold Text Example:}
    \begin{itemize} % Bullet points
        \item First point
        \item Second point
        \item \alert{Important point} (Highlight color)
    \end{itemize}
\end{frame}

```

### Mathematical Slides

You can use `\pause` to reveal content step-by-step.

```latex
\begin{frame}{Math Concept}
    Here is the formula for Convolution:
    \pause % Presentation waits here until you click
    \begin{equation}
        S(i, j) = \sum_{m} \sum_{n} I(i+m, j+n) \cdot K(m, n)
    \end{equation}
    \pause
    \textbf{Meaning:} It extracts features like edges.
\end{frame}

```

### Two-Column Layout (Text + Image/Block)

This is essential for professional slides. Use `\begin{columns}[T]` (Top alignment).

```latex
\begin{frame}{Two Column Layout}
    \begin{columns}[T]
        % --- Left Column ---
        \begin{column}{0.45\textwidth}
            \begin{block}{Left Block Title}
                Content for the left side.
            \end{block}
        \end{column}

        % --- Right Column ---
        \begin{column}{0.55\textwidth}
            \begin{alertblock}{Right Block Title}
                Content for the right side (different color).
            \end{alertblock}
        \end{column}
    \end{columns}
\end{frame}

```

### Adding Images

Always wrap images in a `figure` environment for captions.

```latex
\begin{figure}
    \centering
    % Replace 'filename' with your actual image file (png, jpg, pdf)
    \includegraphics[width=0.8\textwidth]{filename.png}
    \caption{Description of the image}
\end{figure}

```

---

## 5. Customizing the Theme Styles

This specific template allows you to toggle visual elements on the fly between slides.

* **Top Bar Style:**
* `\useThinBar`: Shows a thin colored line at the top.
* `\useThickBar`: Shows a thicker header bar (useful for section transitions).


* **Footer/Logo Style:**
* `\useStyleFull`: Default decoration.
* `\useStyleLogoOnly`: Minimalist, usually just the HUST logo.
* `\useStyleHustFull` / `\useStyleHustOnly`: Variations of the above.



**Example Usage:**

```latex
% Switch to clean layout for a large diagram
\useStyleLogoOnly
\begin{frame}{Large Diagram}
   ...
\end{frame}
% Switch back to full style
\useStyleHustFull

```

---

## 6. Closing the Presentation

### Contact Page

The template includes a specific command to generate a contact slide.

```latex
\hustcontactpage{HEADER TEXT}{
    \textbf{Group:} Your Group Name \\
    \textbf{Email:} email@example.com \\
    \textit{Thank you for listening!}
}

```

### References (Bibliography)

Use `[allowframebreaks]` so references automatically spill over to a new slide if the list is long.

```latex
\begin{frame}[allowframebreaks]{References}
    \begin{thebibliography}{99}
        \bibitem{ref1} Author Name, \emph{Book Title}, Publisher, Year.
        \bibitem{ref2} Author Name, "Paper Title", \emph{Journal}, Year.
    \end{thebibliography}
\end{frame}

```

### Thank You Page

Final closing slide.

```latex
\hustthankyou

```
