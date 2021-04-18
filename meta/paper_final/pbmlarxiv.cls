\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{pbmlarxiv}[2018/04/04 PdfLaTeX class for PBML-like look for arXiv]
\ClassInfo{pbmlarxiv}{$Id: pbmlarxiv.cls 1341 2016-04-04 12:27:19Z popel $\@gobble}

% How to submit a PBML article to arXiv.org?
% - Substitute "\documentclass{pbml}" with "\documentclass{pbmlarxiv} \pdfoutput=1".
% - Rename MAIN.tex to ms.tex (this is needed only if there are multiple *.tex files).
% - Compile with pdflatex (arXiv does not support xelatex).
% - There is no PBML logo, header and footer, but the page breaks and linebreaks should be the same.
% - Submit to arxiv including
%   - pbmlarxiv.cls (this file)
%   - pbmlabbrvnat.bst
%   - all the included files (e.g. pdf figures)
%   - the ms.bbl file
%     BibTeX (*.bib) files are not needed/supported by arXiv.

% Pass all global options to the "article" class
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions
\LoadClass[twoside]{article}

% Paper dimensions -- somewhat smaller than B5
\paperwidth 170mm
\paperheight 240mm

% Sizes (\normalsize and below kept at default)
\def\large{\@setfontsize\large\@xipt{13.5}}
\def\Large{\@setfontsize\Large\@xiipt{14.4}}
\def\LARGE{\@setfontsize\LARGE\@xivpt{18}} % unused
\def\huge{\@setfontsize\huge\@xviipt\@xviipt} % for draft logo
\let\Huge\huge % unused

% Lists
\setlength\partopsep{2\p@ \@plus 1\p@ \@minus 1\p@}
\def\@listi{\leftmargin\leftmargini
            \parsep \z@
            \topsep \z@
            \itemsep \z@}
\let\@listI\@listi
\@listi
\def\@listii {\leftmargin\leftmarginii
              \labelwidth\leftmarginii
              \advance\labelwidth-\labelsep
              \parsep \z@
              \topsep \z@
              \itemsep \z@}
\def\@listiii{\leftmargin\leftmarginiii
              \labelwidth\leftmarginiii
              \advance\labelwidth-\labelsep
              \parsep \z@
              \topsep \z@
              \itemsep \z@}
\def\@listiv {\leftmargin\leftmarginiv
              \labelwidth\leftmarginiv
              \advance\labelwidth-\labelsep}
\def\@listv  {\leftmargin\leftmarginv
              \labelwidth\leftmarginv
              \advance\labelwidth-\labelsep}
\def\@listvi {\leftmargin\leftmarginvi
              \labelwidth\leftmarginvi
              \advance\labelwidth-\labelsep}

% Skips
\bigskipamount\baselineskip
\medskipamount.5\baselineskip
\smallskipamount\medskipamount

\textfloatsep\baselineskip
\intextsep\baselineskip
\floatsep\baselineskip
\@fptop\z@
\@fpsep\baselineskip

% Page layout
\topskip\baselineskip
\parskip \z@
\textwidth 29cc
\evensidemargin 22mm
\oddsidemargin \paperwidth
\advance\oddsidemargin -\textwidth
\advance\oddsidemargin -\evensidemargin
\advance\oddsidemargin -1in
\advance\evensidemargin -1in
\footskip 24\p@
\headheight 12\p@
\headsep 24\p@
\textheight 42\baselineskip
\topmargin 22mm
\advance\topmargin -1in
\parindent 14\p@


% Sectioning, only \section, \subsection and \subsubsection supported
\def\section{\@startsection{section}\@ne\z@{12\p@}{12\p@ minus 6\p@}%
            {\normalfont\large\bfseries}}
\def\pbml@subsection{\@startsection{subsection}\tw@\z@{12\p@}{12\p@ minus 6\p@}%
                    {\normalfont\bfseries}}
\def\pbml@subsubsection{\@startsection{subsubsection}{3}\z@{12\p@}{12\p@ minus 6\p@}%
                       {\normalfont}}
\def\pbml@xsubsection{\@startsection{subsection}\tw@\z@{12\p@}{-.5em}%
                     {\normalfont\bfseries}}
\def\pbml@xsubsubsection{\@startsection{subsubsection}{3}\z@{12\p@}{-.5em}%
                        {\normalfont\bfseries}}

\def\subsection{\@ifstar{\pbml@subsection*}{\pbml@testsec{subsection}}}
\def\subsubsection{\@ifstar{\pbml@subsubsection*}{\pbml@testsec{subsubsection}}}

\def\pbml@testsec#1#2{\ifcat$#2$\@nameuse{pbml@x#1}{}\else\@nameuse{pbml@#1}{#2}\fi}

\setcounter{secnumdepth}{3}
\renewcommand\@seccntformat[1]{\@nameuse{the#1}.\hspace{.5em}}


% Abstract

\renewenvironment{abstract}{\normalfont\normalsize
  \vskip 12\p@ minus 4\p@\par
  \noindent\textbf{Abstract}\par \small}{\par
  \vskip 12\p@ minus 4\p@ \hrule height \p@}

% Captions, always centered
\abovecaptionskip\baselineskip
\belowcaptionskip\baselineskip

\def\@makecaption#1#2{\vskip\abovecaptionskip
  {\centering\normalfont\normalsize\sffamily\itshape #1. #2\par}\vskip\belowcaptionskip}

% Required packages
\RequirePackage{euler}      % math
\RequirePackage{graphicx}   % needed for logo
\RequirePackage{multicol}   % multicolumn text
\RequirePackage{keyval}     % author names and addresses
\RequirePackage{url}        % URL formatting
\RequirePackage{natbib}     % bibliography

% XeLaTeX and amssymb have a conflict with the following five symbols:
\let\Finv\relax
\let\Game\relax
\let\beth\relax
\let\gimel\relax
\let\daleth\relax
\RequirePackage{amssymb}

% xltxtra must be loaded after amsthm
\RequirePackage{amsthm}

% Save the definition of \url

\let\PBMLurl\url

% Make a dummy definition so that we have at least something

\let\HyperrefURL\url

% Slanted -> italics

\let\sl\it
\let\textsl\textit

% Widows and clubs forbidden by default
\widowpenalty= 10000
\clubpenalty= 10000

% Bibliography

\renewenvironment{thebibliography}[1]{%
 \section*{\bibname}\small
 \list
 {}{\setlength{\labelwidth}{0pt}\setlength{\leftmargin}{\parindent}
 \setlength{\itemindent}{-\parindent}}
 \def\newblock{\hskip .11em plus .33em minus -.07em}
 \sloppy\clubpenalty4000\widowpenalty4000
 \sfcode`\.=1000\relax}%
 {\endlist}

% Hacked references, replacing fullname with natbib

\def\cite{\citep}
\def\shortcite{\citeyearpar}
\def\namecite{\citet}
\def\citegood{\citealp}
\def\shortcitegood{\citeyear}
\def\namecitegood{\citealt}

% Cross references, prefix will be defined by \PBMLinclude
\def\pbml@prefix{} % default for single article

% ***************** For hyperref ****************

% \footnote{\url{http://hash.character.com#1}} does not work with hyperref (you would need to escape the #)
% http://tex.stackexchange.com/questions/12855/getting-those-signs-in-the-footnote
% The following solution should work with both \documentclass[color]{pbml} (used by the authors)
% and \PBMLhyperrefexternaltrue written in MAIN.pagerange (used in production).
\newcommand\footnoteurl@internal[2]{\footnote{\url@{#1}}}
\DeclareRobustCommand{\footnoteurl}{\hyper@normalise\footnoteurl@internal}

% Title page of an article, start with defining keys

\newif\ifpbml@aut@corresponding
\define@key{PBMLaut}{surname}{\def\pbml@surname{#1}}
\define@key{PBMLaut}{initials}{\def\pbml@initials{#1~}}
\define@key{PBMLaut}{firstname}{\def\pbml@firstname{#1}}
\define@key{PBMLaut}{institute}{\def\pbml@aut@institute{#1}}
\define@key{PBMLaut}{corresponding}[yes]{% it's invoked by either corresponding,
    \strcfstr{yes}{#1}\ifsame\pbml@aut@correspondingtrue\fi% corresponding=yes,
    \strcfstr{1}{#1}\ifsame\pbml@aut@correspondingtrue\fi} % or corresponding=1
\define@key{PBMLaut}{address}{\def\pbml@aut@address{#1}}
\define@key{PBMLaut}{email}{\def\pbml@aut@email{#1}}

%% by introducing a new institute
%%  - the counter is incremented
%%  - the author's label is asigned a value (1, 2, 3, ... for a, b, c, ...)
%%  - the institute starting with an upper index (a,b,c...) is appended to
%%    token register of institutes
\newcount\authmark\authmark=0   % counter for institutes
\def\institute#1#2{%           \institute{i:ufal}{Institute of F. & App. Ling.}
    \ifcat$\the\pbml@afflist$\else%  if this is not the first institute
        \pbml@addtoaff{\\\affilindex{#1} }\fi% add newline and index (b,c,..)
    \advance\authmark by 1
    \ifnum\authmark=2           % if there is more than one institute =>
        \def\tempindex{\upindex{a}}  % => add index 'a' at the entire begining
        \pbml@afflist\expandafter{\expandafter\tempindex\the\pbml@afflist}\fi
    \expandafter\newcount\csname #1\endcsname%  new counter "\<user's label>"
    \csname #1\endcsname=\authmark
    \pbml@addtoaff{ #2}% append the name of institute to affil. token register
}

%% by introducing a new author
%%  - the author's name is appended to author token register (it is used for
%%    "Cite as" line in the footer of the first page)
%%  - the author's name and the upper indexes of their institutes are appended
%%    to token register of authors&affiliations (it is used for title page)
%%  - the author's name and address is copyied to special variables, if this
%%    author is marked as corresponding author
\def\pbml@csurname{}\def\pbml@cinitials{}\def\pbml@cfirstname{}%
\def\pbml@caut@address{}\def\pbml@caut@email{}%
\def\author#1{%
    \def\pbml@surname{}\def\pbml@initials{}\def\pbml@firstname{}%
    \def\pbml@aut@address{}\def\pbml@aut@email{}%
    \def\pbml@aut@institute{authmark}%
    \pbml@aut@correspondingfalse%
    \setkeys{PBMLaut}{#1}\pbml@compose@autlist%
    \ifpbml@aut@corresponding%
        \let\pbml@csurname\pbml@surname%
        \let\pbml@cinitials\pbml@initials%
        \let\pbml@cfirstname\pbml@firstname%
        \let\pbml@caut@address\pbml@aut@address%
        \let\pbml@caut@email\pbml@aut@email%
    \fi}
\newcommand\testcorrespondingauthor[2]{%
    \ifx\@empty#1
        \typeout{PROBLEM: Missing corresponding author's #2.}
        \def#1{{\bf MISSING} #2}
    \fi}
\newif\ifsame
\newcommand{\strcfstr}[2]{%
  \samefalse
  \begingroup
   \def\1{#1}\def\2{#2}%
   \ifx\1\2 \endgroup\sametrue\else\endgroup\fi}

\def\pbml@compose@autlist{%
    \ifcat$\the\pbml@autlist$\else \pbml@addtoaut{, }\pbml@addtoautaff{, }\fi
    \expandafter\pbml@addtoaut\expandafter{\pbml@firstname~}%
    \expandafter\pbml@addtoaut\expandafter{\pbml@initials}%
    \expandafter\pbml@addtoaut\expandafter{\pbml@surname}%
    \expandafter\pbml@addtoautaff\expandafter{\pbml@firstname~}%
    \expandafter\pbml@addtoautaff\expandafter{\pbml@initials}%
    \expandafter\pbml@addtoautaff\expandafter{\pbml@surname}%
    \expandafter\pbml@addtoautaff\expandafter{%
        \expandafter\segmentaffil\expandafter{\pbml@aut@institute}}
}

% macros for upper index for affiliation
\def\affilindex#1{\upindex{\expandafter\@alph\csname #1\endcsname}}
\def\upindex#1{{$^{#1}$}}
% parsing the affiliation of this kind:  institute={i:ufal,i:utkl}
\def\segmentaffil#1{\firstaffil#1,,} % segmentation of more than one affiliation
\def\firstaffil#1,{%
  \if:#1:\else%
    \ifnum\authmark>1\affilindex{#1}\fi%affilindex only if there is >1 institute
  \expandafter\firstaffil\fi}

%% Token registers for three strings
% 1) Smith, Byron
% 2) Smith^a, Byron^b
% 3) ^a Oxford, ^b Cambridge
\def\pbml@addtoaut#1{\pbml@autlist\expandafter{\the\pbml@autlist #1}}
\newtoks\pbml@autlist       % list of authors
\def\pbml@addtoautaff#1{\pbml@autafflist\expandafter{\the\pbml@autafflist #1}}
\newtoks\pbml@autafflist    % list of authors with references to affiliations
\def\pbml@addtoaff#1{\pbml@afflist\expandafter{\the\pbml@afflist #1}}
\newtoks\pbml@afflist       % list of affiliations (referenced to)

\def\titlelinebreak#1{} % by default \titlelinebreak means nothing

\def\title#{\def\pbml@title}
\def\shorttitle#{\def\pbml@pstitle}
\def\shortauthor#{\def\pbml@psauthor}
\let\shortauthors\shortauthor % alias

% Auxilliary macro for the running head

\def\pbml@def#1 {\expandafter\ifx\csname pbml@ps#1\endcsname\relax
    \expandafter\def\expandafter\pbml@temp\expandafter{\csname pbml@#1\endcsname}%
    \expandafter\let\csname pbml@ps#1\endcsname\pbml@temp
    \fi}

\newdimen\pbml@hwidth
\newbox\pbml@hbox

% Find information in the .aux file and compose the page range

\def\pbml@makerange{%
    \expandafter\ifx\csname\pbml@prefix @@lastpage@@\endcsname\relax
      \expandafter\ifx\csname\pbml@prefix @@firstpage@@\endcsname\relax
        \edef\pbml@pagerange{??–???}%
      \else
        \edef\pbml@pagerange{\@nameuse{\pbml@prefix @@firstpage@@}–???}%
      \fi
    \else
    \expandafter\ifx\csname\pbml@prefix @@firstpage@@\endcsname\relax
      \edef\pbml@pagerange{???–???}% an error has occured
    \else
      \edef\pbml@tempa{\@nameuse{\pbml@prefix @@firstpage@@}}%
      \edef\pbml@tempb{\@nameuse{\pbml@prefix @@lastpage@@}}%
      \ifnum\pbml@tempa=\pbml@tempb
        \edef\pbml@pagerange{\pbml@tempb}%
      \else
        \edef\pbml@pagerange{\pbml@tempa–\pbml@tempb}%
      \fi
    \fi \fi}

% Print the header

\def\pbml@makeheader{\cleardoublepage \normalfont \normalsize
    \pbml@mktitle@warning
    \gdef\thesection{\@arabic\c@section}%
    \setcounter{section}{0}\setcounter{footnote}{0}%
    \setcounter{table}{0}\setcounter{figure}{0}%
    \pbml@makerange
    %\thispagestyle{PBMLempty}%
    \setbox\pbml@hbox \hbox{\sf \large \bfseries}\pbml@hwidth \wd\pbml@hbox
    \vbox to 11mm{}\vskip 5\p@
    \centerline{}\vskip \p@
    \hrule height \z@ depth .5\p@ \vskip 3\p@
    \centerline{\copy\pbml@hbox}\vskip 16\p@}

\def\pbml@mktitle@warning{\def\maketitle{\ClassWarning
    {pbml}{\string\maketitle\space is not allowed here.\MessageBreak
           Your command was ignored and procesing continues.\MessageBreak
           This error was found}}}


\newif\ifpbml@norange

% Make a title of an article
\renewcommand*\maketitle[1][0]{\pbml@makeheader
    \testcorrespondingauthor{\pbml@cfirstname}{first name}
    \testcorrespondingauthor{\pbml@csurname}{surname}
    \testcorrespondingauthor{\pbml@caut@email}{e-mail}
    \testcorrespondingauthor{\pbml@caut@address}{address}
    {\pbml@thanks
      \def\titlelinebreak{\\} % titlelinebreak should do the linebreak here
      \centering \Large \sf {\bfseries \pbml@title\par
      }\vskip 18\p@
      \the\pbml@autafflist\par\vskip 10\p@
      {\scriptsize\the\pbml@afflist\par
       %\vskip 18\p@ Available online XX.YY.ZZZZ\par TODO
      }
    }\vskip 24\p@
    \hrule height \p@
    \vskip #1\baselineskip
    \expandafter\def\expandafter\pbml@author\expandafter{\the\pbml@autlist}
    \pbml@def title \pbml@def author
    \markright{\pbml@psauthor\hfill\pbml@pstitle\ (\pbml@pagerange)}%
    \bibliographystyle{pbmlabbrvnat}%
    \setcitestyle{round,authoryear}%
    \pbml@mkcpr{\\\\\\}

}
\let\PBMLmaketitle\maketitle %this is a hack because both bidi package and pbml class redefine \maketitle


% Macros for footnotes and copyright

\def\pbml@nofootnote{\typeout{\string\footnote\space is not allowed here.^^J%
    Use \string\thanks\space instead.}}

\let\thanks\@gobble
\def\pbml@thanks{\let\footnote\pbml@nofootnote}

\footnotesep\baselineskip

\def\pbml@mkcpr#1{\setbox\pbml@C@box=\vbox{\footnotesize\sffamily
    \noindent\vrule width \z@ height \footnotesep
    #1\par}\pbml@C@ht=\ht\pbml@C@box
    \advance\pbml@C@ht\skip\footins
    \divide\pbml@C@ht\baselineskip
    \multiply\pbml@C@ht\baselineskip
    \advance\pbml@C@ht\baselineskip
    \advance\pbml@C@ht-\footskip
    \expandafter\enlargethispage\expandafter{\expandafter-\the\pbml@C@ht}%
    \advance\pbml@C@ht\footskip}

\newbox\pbml@C@box
\newdimen\pbml@C@ht

\renewcommand\@makefntext[1]{\parindent 14\p@ \noindent \hb@xt@ \parindent{\hss \@makefnmark}#1}


\newcommand{\correspondingaddress}{
  \vspace{5mm}
  \vfill
  \begin{minipage}{\textwidth}
   \textbf{Address for correspondence:}\\
   \pbml@cfirstname~\pbml@cinitials\pbml@csurname\\
   \texttt{\pbml@caut@email}\\
   \pbml@caut@address
  \end{minipage}
}

% For print version use grayscale images, for online use color images if available.
\graphicspath{{./color/}{./grayscale/}}


%% ========== Fonts for both LaTeX and XeLaTeX ===============

\RequirePackage[T1]{fontenc}
\RequirePackage[utf8x]{inputenc}
% Don't warn if fonts are not available - LaTeX wouldn't produce the same output anyway
%\IfFileExists{tgpagella.sty}{\RequirePackage{tgpagella}}{}
%\IfFileExists{tgheros.sty}{\RequirePackage{tgheros}}{}
%\RequirePackage{newpxtext,newpxmath}
\RequirePackage{newpxtext}
\RequirePackage[scaled=0.8]{DejaVuSansMono}
\RequirePackage[scaled=0.83]{DejaVuSans}

\RequirePackage[
   pdfdisplaydoctitle,
   pdfcreator=XeLaTeX\ with\ Hyperref\ package,
   bookmarks, bookmarksopen,
   pdfborderstyle={/S/U/W 1},
   unicode, pdfencoding=auto
 ]{hyperref}

\endinput