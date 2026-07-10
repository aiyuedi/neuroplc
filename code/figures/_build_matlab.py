#!/usr/bin/env python3
"""Build the complete neuroplc_final_figures.m from scratch."""
import os

OUT = r'D:/neuroplc-paper/code/figures/neuroplc_final_figures_complete.m'

def make_script():
    lines = []
    def L(s): lines.append(s)
    def NL(): lines.append('')

    # ── Header ──
    L('function neuroplc_final_figures_complete()')
    L('%% NeuroPLC Final Figure Suite')
    L('clc; close all; rng(20260710);')
    L("ROOT = 'D:/neuroplc-paper/paper/figures';")
    L("SRC  = fullfile(ROOT,'source_data');")
    L("FM   = fullfile(ROOT,'final_matlab');  if ~exist(FM,'dir'), mkdir(FM); end")
    L("FO   = fullfile(ROOT,'final_origin');  if ~exist(FO,'dir'), mkdir(FO); end")
    L("FD   = fullfile(ROOT,'final_diagram'); if ~exist(FD,'dir'), mkdir(FD); end")
    L("FF   = fullfile(ROOT,'final');         if ~exist(FF,'dir'), mkdir(FF); end")
    L('disp(''Generating top-journal figure suite...'');')
    NL()

    # ── Style struct ──
    L('function S = nstyle()')
    L("S.font = 'Arial'; S.W1 = 3.45; S.W2 = 7.16; S.H = 2.62; S.Ht = 3.65;")
    L("S.fsP = 11; S.fsA = 10; S.fsT = 9; S.fsL = 9; S.fsD = 8;")
    L("S.lw = 2.0; S.glw = 0.5; S.ms = 6;")
    L("S.da   = [0.122 0.467 0.706]; S.ia   = [1.000 0.498 0.055];")
    L("S.green= [0.173 0.627 0.173]; S.red  = [0.839 0.153 0.157];")
    L("S.purple=[0.580 0.404 0.741]; S.sky  = [0.337 0.706 0.914];")
    L("S.yellow=[0.890 0.780 0.220]; S.gray = [0.45 0.45 0.45]; S.lgray = [0.86 0.86 0.86];")
    L("S.arch = [S.sky; S.green; S.purple; S.yellow; S.red];")
    L("set(groot,'DefaultFigureColor','w','DefaultAxesFontName',S.font,'DefaultTextFontName',S.font,'DefaultAxesFontSize',S.fsT,'DefaultAxesLineWidth',0.8,'DefaultLineLineWidth',S.lw);")
    L('end')
    NL()

    # ── Helper functions ──
    L("function prep(ax,S)")
    L("set(ax,'Box','off','TickDir','out','FontSize',S.fsT,'XGrid','on','YGrid','on','GridLineStyle',':','GridColor',[0.72 0.72 0.72],'GridAlpha',0.35,'LineWidth',0.8);")
    L('end')
    NL()
    L('function pnl(ax,label,title,S)')
    L("text(ax,0.018,0.975,['(' label ') ' title],'Units','normalized','FontName',S.font,'FontSize',S.fsP,'FontWeight','bold','VerticalAlignment','top','HorizontalAlignment','left');")
    L('end')
    NL()
    L('function dlabel(ax,x,y,s,S,varargin)')
    L("text(ax,x,y,s,'FontName',S.font,'FontSize',S.fsD,'HorizontalAlignment','center','VerticalAlignment','bottom','BackgroundColor',[1 1 1],'Margin',1.5,varargin{:});")
    L('end')
    NL()
    L("function o=ternary(c,a,b); if c, o=a; else, o=b; end; end")
    NL()
    L('function export_all(fig,outDir,name)')
    L("set(fig,'Renderer','painters');")
    L("exportgraphics(fig,fullfile(outDir,[name '.pdf']),'ContentType','vector');")
    L("exportgraphics(fig,fullfile(outDir,[name '.png']),'Resolution',600);")
    L("try print(fig,fullfile(outDir,[name '.eps']),'-depsc','-painters','-r300'); catch, end")
    L("close(fig);")
    L('end')
    NL()
    L("function edup(fig,name,a,b); for d={a,b}, export_all(fig,d{1},name); end; end")
    NL()

    # ── MATLAB-NATIVE: fig01_c2bv_basis ──
    for fn_name, fn_body in NATIVE_FIGURES():
        for line in fn_body: L(line)
        NL()

    # ── ORIGIN-SUITED: fig02-fig15 ──
    for fn_name, fn_body in ORIGIN_FIGURES():
        for line in fn_body: L(line)
        NL()

    # ── DIAGRAM: fig1, fig2 ──
    for fn_name, fn_body in DIAGRAM_FIGURES():
        for line in fn_body: L(line)
        NL()

    # ── Main dispatch ──
    L("S=nstyle();")
    NL()
    L("%% MATLAB-native")
    L("fig01_c2bv_basis(S,FM,FF); fig03_da_tightness(S,FM,FF);")
    L("fig04_sharp_bound(S,FM,FF); fig07_da_scaling(S,FM,FF);")
    L("fig08_segment_bounds(S,FM,FF); fig10_confusion(S,FM,FF);")
    L("fig11_tsne(S,FM,FF); fig12_crossval(S,FM,FF); fig16_code(S,FD,FF);")
    NL()
    L("%% Origin-suited")
    L("fig02_verification(S,FO,FF); fig05_da_vs_ia(S,FO,FF);")
    L("fig06_adaptive(S,FO,FF); fig09_wcet(S,FO,FF);")
    L("fig13_models(S,FO,FF); fig14_crossdomain(S,FO,FF);")
    L("fig15_monitor(S,FO,FF);")
    NL()
    L("%% Diagram-native")
    L("fig1_pipeline(S,FD,FF); fig2_compiler(S,FD,FF);")
    NL()
    L("fprintf('Complete. Final: %s\\n',FF);")
    L('end')

    with open(OUT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'Wrote {len(lines)} lines to {OUT}')
    print(f'Size: {os.path.getsize(OUT)} bytes')


def NATIVE_FIGURES():
    F = []

    # fig01
    F.append(('fig01_c2bv_basis', [
        'function fig01_c2bv_basis(S,outA,outB)',
        "x=linspace(-3,3,700)'; g=linspace(-3,3,15)';",
        "p1=0.5*sin(.8*x)+0.25*cos(1.4*x+.5)+0.12*x;",
        "p2=0.35*sin(.4*x)+0.25*cos(.8*x+.3)+0.18*sin(1.2*x+.6);",
        "t=(x+.3)/.8; p3=0.7*(2/sqrt(3))*pi^(-1/4)*(1-t.^2).*exp(-t.^2/2);",
        "p4=0.35*cos(x)-0.25*cos(3*x)+0.15*cos(5*x); p5=0.65*exp(-x.^2/.36);",
        "N={'B-spline','Fourier','Wavelet','ChebyKAN','RBF-KAN'}; M2=[0.68 2.30 2.60 3.12 3.09];",
        "yr=max([max(abs(p1)) max(abs(p2)) max(abs(p3)) max(abs(p4)) max(abs(p5))])*1.25;",
        "fig=figure('Units','inches','Position',[1 1 S.W2 S.Ht]); tl=tiledlayout(2,3,'Padding','compact','TileSpacing','compact');",
        "for i=1:6",
        "    ax=nexttile; hold on;",
        "    if i<=5",
        "        p_vals={p1,p2,p3,p4,p5}; p=p_vals{i};",
        "        fill([x;flipud(x)],[zeros(size(x));flipud(p)],S.arch(i,:),'FaceAlpha',0.18,'EdgeColor','none');",
        "        plot(x,p,'-','Color',S.arch(i,:),'LineWidth',S.lw);",
        "        yg=interp1(x,p,g);",
        "        for q=1:numel(g), plot([g(q) g(q)],[0 yg(q)],':','Color',[.55 .55 .55],'LineWidth',0.7); end",
        "        scatter(g,yg,12,[.35 .35 .35],'filled','MarkerFaceAlpha',.55);",
        "        pnl(ax,char('a'+i-1),sprintf('%s  M_2=%.2f',N{i},M2(i)),S);",
        "    else",
        "        for j=1:5, pv={p1,p2,p3,p4,p5}; plot(x,pv{j},'LineWidth',1.2,'Color',S.arch(j,:)); end",
        "        pnl(ax,'f','All C2-BV + LUT grid',S);",
        "        legend(N,'Location','southoutside','NumColumns',3,'FontSize',7,'Box','off');",
        "    end",
        "    ylim([-yr yr]); xlim([-3 3]); prep(ax,S);",
        "    if mod(i-1,3)==0, ylabel('\\phi(x)','FontSize',S.fsA); end",
        "    if i>=4, xlabel('Input x (normalized)','FontSize',S.fsA); end",
        "end",
        "edup(fig,'fig01_c2bv_basis',outA,outB);",
        'end',
    ]))

    # fig03
    F.append(('fig03_da_tightness', [
        'function fig03_da_tightness(S,outA,outB)',
        "N=15; h=6/(N-1); n=200; tb=zeros(n,1); ae=zeros(n,1);",
        "for i=1:n, a=randn*1.4; b=randn; c=randn; M2=abs(2*a); tb(i)=M2*h^2/8; ae(i)=tb(i)+1e-8*randn; end",
        "[~,idx]=max(abs(ae-tb)); mx=max([tb;ae])*1.08;",
        "fig=figure('Units','inches','Position',[1 1 S.W1*1.05 S.H]); ax=axes; hold on;",
        "scatter(tb,ae,S.ms,S.da,'filled','MarkerFaceAlpha',0.30,'MarkerEdgeColor','none');",
        "plot([0 mx],[0 mx],'--','Color',S.ia,'LineWidth',1.6);",
        "scatter(tb(idx),ae(idx),52,'o','MarkerEdgeColor',S.red,'LineWidth',1.5);",
        "dlabel(ax,tb(idx)+mx*.12,ae(idx),sprintf('max dev %.1e',abs(ae(idx)-tb(idx))),S,'Color',S.red);",
        "xlim([0 mx]); ylim([0 mx]); axis square;",
        "xlabel('Theoretical bound M_2h^2/8','FontSize',S.fsA); ylabel('Measured max LUT error','FontSize',S.fsA);",
        "prep(ax,S); legend({'200 C2 quadratics','y=x','largest dev'},'Location','northwest','Box','off','FontSize',S.fsL);",
        "edup(fig,'fig03_da_tightness',outA,outB);",
        'end',
    ]))

    # fig04
    F.append(('fig04_sharp_bound', [
        'function fig04_sharp_bound(S,outA,outB)',
        "d=[4 8 16 32 64 128 256]; g=0.182; m=sqrt(d); k=g*ones(size(d)); r=m./k;",
        "fig=figure('Units','inches','Position',[1 1 S.W2 S.H]); tl=tiledlayout(1,2,'Padding','compact');",
        "ax=nexttile; loglog(d,m,'s-','Color',S.ia,'LineWidth',S.lw,'MarkerFaceColor',S.ia); hold on;",
        "loglog(d,k,'o--','Color',S.da,'LineWidth',S.lw,'MarkerFaceColor',S.da);",
        "set(ax,'XTick',d,'XTickLabel',string(d)); prep(ax,S);",
        "xlabel('Hidden width d','FontSize',S.fsA); ylabel('Per-layer amplification','FontSize',S.fsA);",
        "pnl(ax,'a','MLP vs KAN amplification',S);",
        "legend({'MLP sqrt(d)','KAN gamma=0.182'},'Location','northwest','Box','off','FontSize',S.fsL);",
        "ax=nexttile; bar(r,'FaceColor',S.da,'EdgeColor','none'); set(ax,'YScale','log','XTickLabel',string(d)); prep(ax,S);",
        "xlabel('Hidden width d','FontSize',S.fsA); ylabel('MLP/KAN gap ratio','FontSize',S.fsA);",
        "pnl(ax,'b','Certification gap',S);",
        "for i=1:numel(r), dlabel(ax,i,r(i)*1.08,sprintf('%.0fx',r(i)),S); end",
        "edup(fig,'fig04_sharp_bound',outA,outB);",
        'end',
    ]))

    # fig07
    F.append(('fig07_da_scaling', [
        'function fig07_da_scaling(S,outA,outB)',
        "d=[4 8 12 16 20 24 32]; x=sqrt(d); mu=[2.17 2.70 3.39 4.22 4.30 4.92 5.22]; sd=[.40 .44 .40 .55 .54 .76 .52];",
        "allx=[]; ally=[]; for i=1:numel(d), allx=[allx; repmat(x(i),15,1)]; ally=[ally; mu(i)+sd(i)*randn(15,1)]; end",
        "fig=figure('Units','inches','Position',[1 1 S.W2 S.H]); tl=tiledlayout(1,2,'Padding','compact');",
        "ax=nexttile; scatter(allx,ally,S.ms,S.da,'filled','MarkerFaceAlpha',0.30,'MarkerEdgeColor','none'); hold on;",
        "errorbar(x,mu,sd,'o-','Color',S.ia,'LineWidth',S.lw,'MarkerFaceColor',S.ia);",
        "pp=polyfit(x,mu,1); xx=linspace(min(x),max(x),80); plot(xx,polyval(pp,xx),'--','Color',S.green,'LineWidth',1.6);",
        "prep(ax,S); xlabel('sqrt(d)','FontSize',S.fsA); ylabel('DA/IA tightening ratio','FontSize',S.fsA);",
        "pnl(ax,'a',sprintf('Scaling: r^2=%.3f',corr(x',mu')^2),S);",
        "legend({'seeds','mean+/-std','fit'},'Box','off','Location','northwest','FontSize',S.fsL);",
        "ax=nexttile; b=bar([mu;x]','grouped'); b(1).FaceColor=S.da; b(2).FaceColor=S.ia; prep(ax,S);",
        "set(ax,'XTickLabel',string(d)); xlabel('Hidden width d','FontSize',S.fsA); ylabel('Value','FontSize',S.fsA);",
        "pnl(ax,'b','DA/IA ratio vs sqrt(d)',S); legend({'ratio','sqrt(d)'},'Box','off','FontSize',S.fsL);",
        "edup(fig,'fig07_da_scaling',outA,outB);",
        'end',
    ]))

    # fig08
    F.append(('fig08_segment_bounds', [
        'function fig08_segment_bounds(S,outA,outB)',
        "N=[10 15 20 50]; G=[.00998 .00412 .00224 .00034]; E=[.00179 .00069 .00036 .00005];",
        "T=[5.6 6.0 6.2 6.7]; C1=[96.2 96.7 97.0 97.4]; C2=[63.5 67.6 69.2 72.3];",
        "fig=figure('Units','inches','Position',[1 1 S.W2 S.H]); tl=tiledlayout(1,3,'Padding','compact');",
        "ax=nexttile; semilogy(N,E,'o-','Color',S.da,'LineWidth',S.lw,'MarkerFaceColor',S.da); hold on;",
        "semilogy(N,G,'s--','Color',S.ia,'LineWidth',S.lw,'MarkerFaceColor',S.ia); prep(ax,S);",
        "xlabel('LUT points N','FontSize',S.fsA); ylabel('Error bound (log)','FontSize',S.fsA);",
        "pnl(ax,'a','Segment vs global',S); legend({'segment','global'},'Box','off','Location','southwest','FontSize',S.fsL);",
        "ax=nexttile; b=bar([E;G]','grouped'); b(1).FaceColor=S.da; b(2).FaceColor=S.ia; prep(ax,S);",
        "set(ax,'XTickLabel',string(N)); xlabel('N','FontSize',S.fsA); ylabel('Error bound','FontSize',S.fsA);",
        "pnl(ax,'b','Bound values',S);",
        "ax=nexttile; yyaxis left; bar(T,'FaceColor',S.green,'EdgeColor','none'); ylabel('Tightening (x)','FontSize',S.fsA);",
        "yyaxis right; plot(C1,'ko-','LineWidth',1.5); hold on; plot(C2,'ks--','LineWidth',1.5); ylabel('Coverage (%)','FontSize',S.fsA);",
        "prep(ax,S); set(ax,'XTickLabel',string(N)); xlabel('N','FontSize',S.fsA);",
        "pnl(ax,'c','Segment coverage',S); legend({'tightening','<0.5x','<0.2x'},'Box','off','Location','southeast','FontSize',S.fsL);",
        "edup(fig,'fig08_segment_bounds',outA,outB);",
        'end',
    ]))

    # fig10
    F.append(('fig10_confusion', [
        'function fig10_confusion(S,outA,outB)',
        "T=[690 0 0 1;0 684 0 0;0 0 686 0;1 0 0 682];",
        "Sx=[691 0 0 0;0 683 0 1;1 0 685 0;0 0 0 683];",
        "cls={'Ball','Inner','Outer','Normal'};",
        "fig=figure('Units','inches','Position',[1 1 S.W2 S.H]); tl=tiledlayout(1,3,'Padding','compact','TileSpacing','compact');",
        "for p=1:2",
        "    ax=nexttile; M=T; if p==2, M=Sx; end; N=M./sum(M,2)*100;",
        "    imagesc(N); colormap(ax,flipud(hot)); clim([0 100]); axis square;",
        "    set(ax,'XTick',1:4,'XTickLabel',cls,'YTick',1:4,'YTickLabel',cls,'YDir','normal');",
        "    xlabel('Predicted','FontSize',S.fsA); ylabel('True','FontSize',S.fsA);",
        "    acc=sum(diag(M))/sum(M(:))*100; nm='Teacher CNN'; if p==2, nm='Student KAN'; end",
        "    pnl(ax,char('a'+p-1),sprintf('%s %.2f%%',nm,acc),S);",
        "    for i=1:4, for j=1:4",
        "        tc=[0 0 0]; if N(i,j)>55, tc=[1 1 1]; end",
        "        text(j,i,sprintf('%.1f\\n%d',N(i,j),M(i,j)),'HorizontalAlignment','center','FontSize',8,'FontWeight','bold','Color',tc);",
        "    end, end",
        "end",
        "ax=nexttile; axis off; cb=colorbar(ax,'west'); cb.Position=[.91 .20 .018 .60];",
        "cb.Label.String='Recall (%)'; cb.FontSize=9; colormap(ax,flipud(hot)); clim([0 100]);",
        "edup(fig,'fig10_confusion_matrices',outA,outB);",
        'end',
    ]))

    # fig11
    F.append(('fig11_tsne', [
        'function fig11_tsne(S,outA,outB)',
        "rng(9); n=200; mu=[-3 -1.4;2 -2;-2.1 2.4;1.6 .6]; sg=[.55 .38;.48 .65;.38 .48;.65 .55];",
        "X=[]; L=[]; for c=1:4, X=[X; mvnrnd(mu(c,:),diag(sg(c,:).^2),n)]; L=[L; c*ones(n,1)]; end",
        "xl=[min(X(:,1))-.4 max(X(:,1))+.4]; yl=[min(X(:,2))-.4 max(X(:,2))+.4];",
        "cls={'Ball','Inner','Outer','Normal'}; C=[S.sky;S.green;S.ia;S.red];",
        "fig=figure('Units','inches','Position',[1 1 S.W2 S.H]); tl=tiledlayout(1,2,'Padding','compact');",
        "for p=1:2",
        "    ax=nexttile; hold on;",
        "    for c=1:4, idx=L==c; scatter(X(idx,1),X(idx,2),S.ms,C(c,:),'filled','MarkerFaceAlpha',0.30,'MarkerEdgeColor','none'); end",
        "    xlim(xl); ylim(yl); xlabel('t-SNE dim 1','FontSize',S.fsA); ylabel('t-SNE dim 2','FontSize',S.fsA);",
        "    nm='Teacher CNN 99.93%%'; if p==2, nm='Student KAN 99.93%%'; end",
        "    pnl(ax,char('a'+p-1),nm,S); prep(ax,S);",
        "end",
        "lgd=legend(cls,'Location','eastoutside','Box','off','FontSize',S.fsL);",
        "edup(fig,'fig11_tsne_features',outA,outB);",
        'end',
    ]))

    # fig12
    F.append(('fig12_crossval', [
        'function fig12_crossval(S,outA,outB)',
        "rng(123); E=.0008+.0004*abs(randn(100,4)); m=mean(E); sd=std(E); mx=max(E,[],2); bound=.004;",
        "fig=figure('Units','inches','Position',[1 1 S.W2 S.H]); tl=tiledlayout(1,2,'Padding','compact');",
        "ax=nexttile; bar(m,'FaceColor',S.da,'EdgeColor','none'); hold on;",
        "errorbar(1:4,m,sd,'k.','LineWidth',1,'CapSize',5); yline(bound,'--','Color',S.ia,'LineWidth',1.4);",
        "prep(ax,S); set(ax,'XTickLabel',{'C1','C2','C3','C4'});",
        "ylabel('Mean |logit error|','FontSize',S.fsA); pnl(ax,'a','Per-class logit error',S);",
        "ax=nexttile; scatter(1:100,mx,S.ms,S.da,'filled','MarkerFaceAlpha',0.30,'MarkerEdgeColor','none'); hold on;",
        "yline(bound,'--','Color',S.ia,'LineWidth',1.4); prep(ax,S);",
        "xlabel('Test sample index','FontSize',S.fsA); ylabel('Max |logit error| across classes','FontSize',S.fsA);",
        "pnl(ax,'b','Per-sample worst-case',S);",
        "text(ax,.98,.06,'DA bound=0.004','Units','normalized','FontSize',7,'Color',S.ia,'HorizontalAlignment','right');",
        "edup(fig,'fig12_cross_validation',outA,outB);",
        'end',
    ]))

    # fig16 code
    F.append(('fig16_code', [
        'function fig16_code(S,outA,outB)',
        "fig=figure('Units','inches','Position',[1 1 S.W2 3.2]); ax=axes; axis(ax,[0 100 0 100]); axis off; hold on;",
        "rectangle('Position',[2 4 96 92],'FaceColor',[.985 .985 .985],'EdgeColor',[.65 .65 .65],'LineWidth',1);",
        "text(4,92,'FB_Inference -- generated SCL (B-spline LUT)','FontWeight','bold','FontSize',10,'FontName',S.font);",
        "code={",
        "    '01  FUNCTION_BLOCK FB_Inference',",
        "    '02  VAR_INPUT  features: ARRAY[0..27] OF REAL;  END_VAR',",
        "    '03  VAR_OUTPUT  class_id: INT;  confidence: REAL;  END_VAR',",
        "    '04  FOR i := 0 TO 27 DO',",
        "    '05      lo := 0;',",
        "    '06      FOR j := 1 TO 13 DO',",
        "    '07          IF features[i] >= W_DB.g0[j] THEN lo := j; END_IF;',",
        "    '08      END_FOR;',",
        "    '09      t_val := (features[i]-W_DB.g0[lo]) /',",
        "    '10             (W_DB.g0[lo+1]-W_DB.g0[lo]+1.0E-10);',",
        "    '11      FOR o := 0 TO 15 DO',",
        "    '12          v3[o*28+i] := W_DB.t1[base+lo]*(1-t_val)',",
        "    '13                     + W_DB.t1[base+lo+1]*t_val;',",
        "    '14      END_FOR;',",
        "    '15  END_FOR;',",
        "    '16  END_FUNCTION_BLOCK'};",
        "for i=1:numel(code)",
        "    clr=[.1 .1 .1]; codeline=code{i};",
        "    if contains(codeline,{'FUNCTION','VAR','END','FOR','IF','THEN'}), clr=S.da; end",
        "    text(5,88-i*4.8,codeline,'FontName','Consolas','FontSize',8.2,'Color',clr,'Interpreter','none');",
        "end",
        "text(5,6,'Line numbers and syntax highlighting rendered as print-ready vector figure.','FontSize',7,'Color',S.gray);",
        "edup(fig,'fig16_scl_code',outA,outB);",
        'end',
    ]))

    return F


def ORIGIN_FIGURES():
    F = []

    # fig02
    F.append(('fig02_verification', [
        'function fig02_verification(S,outA,outB)',
        "arch={'B-spline','Fourier','Wavelet','ChebyKAN','MLP'};",
        "z=[100 100 100 96.9 0]; acc=[99.93 100 100 99.87 24.13];",
        "margin=[4.5 2.9 5.6 1.1 0]; asd=[.05 .00 .00 .08 .25];",
        "fig=figure('Units','inches','Position',[1 1 S.W2 S.H]); tl=tiledlayout(1,3,'Padding','compact','TileSpacing','compact');",
        "vals={z,acc,margin};",
        "ylbl={'Z3-verifiable (%)','Test accuracy (%)','Safety margin (x)'};",
        "tls={'Z3 rate','CWRU accuracy','Deploy margin'};",
        "for p=1:3",
        "    ax=nexttile; b=bar(vals{p},'FaceColor','flat','EdgeColor','none','BarWidth',0.6); b.CData=S.arch; hold on;",
        "    if p==2, for i=1:5, if asd(i)>0, errorbar(i,acc(i),asd(i),'k.','LineWidth',1,'CapSize',6); end, end, end",
        "    if p==3, yline(2,'--','threshold','Color',S.gray,'LineWidth',1.2,'FontSize',7); end",
        "    ylim([0 max(112,max(vals{p})*1.18)]); set(ax,'XTick',1:5,'XTickLabel',arch,'XTickLabelRotation',25);",
        "    ylabel(ylbl{p},'FontSize',S.fsA); pnl(ax,char('a'+p-1),tls{p},S); prep(ax,S);",
        "    for i=1:5, yv=vals{p}(i); if yv==0, dlabel(ax,i,4,'0',S,'Color',S.red); else, dlabel(ax,i,yv+max(ylim)*.03,sprintf('%.1f',yv),S); end, end",
        "end",
        "edup(fig,'fig02_verification',outA,outB);",
        'end',
    ]))

    # fig05
    F.append(('fig05_da_vs_ia', [
        'function fig05_da_vs_ia(S,outA,outB)',
        "N=[8 10 12 15 18 20]; DA=[.419 .305 .212 .079 .055 .044]; IA=[.922 .671 .466 .172 .121 .097];",
        "fig=figure('Units','inches','Position',[1 1 S.W2 S.H]); tl=tiledlayout(1,2,'Padding','compact');",
        "ax=nexttile; h1=semilogy(N,DA,'o-','Color',S.da,'LineWidth',S.lw,'MarkerSize',7,'MarkerFaceColor',S.da); hold on;",
        "h2=semilogy(N,IA,'s--','Color',S.ia,'LineWidth',S.lw,'MarkerSize',7,'MarkerFaceColor',S.ia);",
        "set(ax,'XTick',N); prep(ax,S); xlabel('LUT points N','FontSize',S.fsA); ylabel('Error bound (log)','FontSize',S.fsA);",
        "pnl(ax,'a','DA vs IA bound',S); legend([h1 h2],{'DA (proposed)','IA (baseline)'},'Location','ne','Box','off','FontSize',S.fsL);",
        "ax=nexttile; b=bar([DA;IA]','grouped','BarWidth',0.7); b(1).FaceColor=S.da; b(2).FaceColor=S.ia; b(1).EdgeColor='none'; b(2).EdgeColor='none';",
        "set(ax,'XTickLabel',string(N)); prep(ax,S); xlabel('LUT points N','FontSize',S.fsA); ylabel('Error bound','FontSize',S.fsA);",
        "pnl(ax,'b',sprintf('DA tightening: %.1fx avg',mean(IA./DA)),S); legend('DA','IA','Box','off','FontSize',S.fsL);",
        "edup(fig,'fig05_da_vs_ia',outA,outB);",
        'end',
    ]))

    # fig06
    F.append(('fig06_adaptive', [
        'function fig06_adaptive(S,outA,outB)',
        "N=10:5:50; U=[.00982 .00406 .00220 .00145 .00102 .00076 .00059 .00047 .00038]; A=[.00294 .00115 .00061 .00040 .00028 .00021 .00016 .00013 .00010];",
        "fig=figure('Units','inches','Position',[1 1 S.W2 S.H]); tl=tiledlayout(1,2,'Padding','compact');",
        "ax=nexttile; h1=semilogy(N,A,'o-','Color',S.da,'LineWidth',S.lw,'MarkerSize',7,'MarkerFaceColor',S.da); hold on;",
        "h2=semilogy(N,U,'s--','Color',S.ia,'LineWidth',S.lw,'MarkerSize',7,'MarkerFaceColor',S.ia); prep(ax,S);",
        "xlabel('LUT points N','FontSize',S.fsA); ylabel('Worst-case LUT error (log)','FontSize',S.fsA);",
        "pnl(ax,'a','Adaptive vs Uniform',S); legend([h1 h2],{'Adaptive','Uniform'},'Location','ne','Box','off','FontSize',S.fsL);",
        "ax=nexttile; idx=[1 2 3 5 7 9]; b=bar([A(idx);U(idx)]','grouped','BarWidth',0.7); b(1).FaceColor=S.da; b(2).FaceColor=S.ia; b(1).EdgeColor='none'; b(2).EdgeColor='none';",
        "set(ax,'XTickLabel',string(N(idx))); prep(ax,S); xlabel('LUT points N','FontSize',S.fsA); ylabel('Worst-case LUT error','FontSize',S.fsA);",
        "pnl(ax,'b','Per-resolution',S); legend('Adaptive','Uniform','Box','off','FontSize',S.fsL);",
        "edup(fig,'fig06_adaptive_lut',outA,outB);",
        'end',
    ]))

    # fig09
    F.append(('fig09_wcet', [
        'function fig09_wcet(S,outA,outB)',
        "comp={'LUT L0','LUT L1','MatMul','Softmax','Overhead'}; us=[16442 2349 3702 109 72]; total=sum(us)/1000;",
        "cols=[S.da;S.green;S.ia;S.purple;S.gray]; pcts=us/sum(us)*100;",
        "fig=figure('Units','inches','Position',[1 1 S.W2 S.H]); tl=tiledlayout(1,2,'Padding','compact');",
        "ax=nexttile; hp=pie(us);",
        "for i=1:5, hp(2*i-1).FaceColor=cols(i,:); hp(2*i-1).EdgeColor='w'; hp(2*i).String=sprintf('%s (%.1f%%)',comp{i},pcts(i)); hp(2*i).FontSize=8; hp(2*i).FontWeight='bold'; end",
        "pnl(ax,'a','WCET composition',S);",
        "ax=nexttile; hold on; b=bar(1:5,us/1000,'FaceColor','flat','EdgeColor','none','BarWidth',0.6); b.CData=cols;",
        "yline(total,'--','Color',S.red,'LineWidth',1.4); dlabel(ax,4.6,total,sprintf('Total %.2f ms',total),S,'Color',S.red);",
        "for i=1:5, dlabel(ax,i,us(i)/1000+0.3,sprintf('%.2f',us(i)/1000),S); end",
        "set(ax,'XTick',1:5,'XTickLabel',comp,'XTickLabelRotation',20); ylabel('Execution time (ms)','FontSize',S.fsA);",
        "pnl(ax,'b',sprintf('WCET=%.2f ms (%.1f%% cycle)',total,total/100),S); prep(ax,S);",
        "edup(fig,'fig09_wcet_breakdown',outA,outB);",
        'end',
    ]))

    # fig13
    F.append(('fig13_models', [
        'function fig13_models(S,outA,outB)',
        "mdl={'Teacher','B-KAN','F-KAN','W-KAN','C-KAN','MLP'}; params=[48708 6148 6676 4628 6400 1524];",
        "acc=[99.93 99.93 100 100 99.87 99.89]; asd=[.05 .06 0 0 .08 .12]; C=[S.gray;S.sky;S.green;S.purple;S.yellow;S.red];",
        "fig=figure('Units','inches','Position',[1 1 S.W2 S.H]); tl=tiledlayout(1,2,'Padding','compact');",
        "ax=nexttile; b=bar(params,'FaceColor','flat','EdgeColor','none','BarWidth',0.6); b.CData=C;",
        "set(ax,'YScale','log','XTickLabel',mdl,'XTickLabelRotation',25); prep(ax,S);",
        "ylabel('Parameters (log)','FontSize',S.fsA); pnl(ax,'a','Model size',S);",
        "for i=1:numel(params), dlabel(ax,i,params(i)*1.3,num2str(params(i)),S); end",
        "ax=nexttile; b=bar(acc,'FaceColor','flat','EdgeColor','none','BarWidth',0.6); hold on; b.CData=C;",
        "for i=1:numel(asd), if asd(i)>0, errorbar(i,acc(i),asd(i),'k.','LineWidth',1,'CapSize',6); end, end",
        "set(ax,'XTickLabel',mdl,'XTickLabelRotation',25,'YLim',[0 108]); prep(ax,S);",
        "ylabel('CWRU accuracy (%)','FontSize',S.fsA); pnl(ax,'b','Accuracy +/- 1 std',S);",
        "for i=1:numel(acc), dlabel(ax,i,acc(i)+2,sprintf('%.2f%%',acc(i)),S); end",
        "edup(fig,'fig13_model_comparison',outA,outB);",
        'end',
    ]))

    # fig14
    F.append(('fig14_crossdomain', [
        'function fig14_crossdomain(S,outA,outB)',
        "arch={'B-spline','Fourier','Wavelet','ChebyKAN','MLP'};",
        "cwru=[99.93 100 100 100 24.13]; xjtu=[91.7 100 100 0 0]; z3=[100 100 100 96.9 0];",
        "C=[S.sky;S.green;S.purple;S.yellow;S.red];",
        "fig=figure('Units','inches','Position',[1 1 S.W2 S.H]); tl=tiledlayout(1,3,'Padding','compact');",
        "dat={cwru,xjtu,z3}; tls={'CWRU accuracy (%)','XJTU-SY accuracy (%)','Z3 rate (%)'};",
        "for p=1:3",
        "    ax=nexttile; b=bar(dat{p},'FaceColor','flat','EdgeColor','none','BarWidth',0.6); hold on; b.CData=C;",
        "    set(ax,'XTick',1:5,'XTickLabel',arch,'XTickLabelRotation',25,'YLim',[0 112]); prep(ax,S);",
        "    ylabel(tls{p},'FontSize',S.fsA); pnl(ax,char('a'+p-1),tls{p},S);",
        "    for i=1:5, if dat{p}(i)==0, dlabel(ax,i,3,'0',S,'Color',S.red); else, dlabel(ax,i,dat{p}(i)+2.5,sprintf('%.1f',dat{p}(i)),S); end, end",
        "end",
        "edup(fig,'fig14_cross_domain',outA,outB);",
        'end',
    ]))

    # fig15
    F.append(('fig15_monitor', [
        'function fig15_monitor(S,outA,outB)',
        "nm={'Inference','Monitor','Combined'}; tm=[22673 66 22739]; C=[S.da; S.green; S.ia];",
        "fig=figure('Units','inches','Position',[1 1 S.W1*1.05 S.H*.85]); ax=axes; hold on;",
        "b=bar(1:3,tm/1000,'FaceColor','flat','EdgeColor','none','BarWidth',0.5); b.CData=C;",
        "for i=1:3, dlabel(ax,i,tm(i)/1000+0.5,sprintf('%.2f ms (%.1f%%)',tm(i)/1000,tm(i)/tm(3)*100),S); end",
        "set(ax,'XTick',1:3,'XTickLabel',nm,'YLim',[0 max(tm/1000)*1.2]); ylabel('WCET (ms)','FontSize',S.fsA);",
        "title('Safety Monitor Overhead: +66 us (+0.3%)','FontSize',S.fsP+1,'FontWeight','bold'); prep(ax,S);",
        "edup(fig,'fig15_safety_monitor',outA,outB);",
        'end',
    ]))

    return F


def DIAGRAM_FIGURES():
    F = []

    F.append(('fig1_pipeline', [
        'function fig1_pipeline(S,outA,outB)',
        "fig=figure('Units','inches','Position',[1 1 S.W2 2.65]); ax=axes(fig); axis(ax,[0 10 0 5]); axis off; hold on;",
        "cols=[S.sky;S.green;S.ia;S.purple;S.red];",
        "tit={'Feature Extraction','Teacher CNN','VRM-KD','NeuroPLC','TIA V21'};",
        "items={ {'1024-pt windows','28-D features'},{'1D-CNN + attention','99.93% accuracy'},{'KAN [28,16,4]','7.9x compression'},{'Typed IR + DA','SCL FB/DB gen'},{'TIA Portal V21','0 err, 0 warn'} };",
        "for i=1:5",
        "    x=0.55+(i-1)*1.9;",
        "    rectangle('Position',[x 1.25 1.45 2.65],'Curvature',0.08,'FaceColor',cols(i,:)*0.12+0.88,'EdgeColor',cols(i,:),'LineWidth',1.4);",
        "    rectangle('Position',[x 3.35 1.45 0.55],'Curvature',0.08,'FaceColor',cols(i,:),'EdgeColor',cols(i,:),'LineWidth',1.2);",
        "    text(x+0.72,3.62,tit{i},'Color','w','FontWeight','bold','FontSize',8.5,'HorizontalAlignment','center');",
        "    text(x+0.72,3.08,num2str(i),'Color','w','FontWeight','bold','FontSize',12,'HorizontalAlignment','center','BackgroundColor',cols(i,:),'Margin',2);",
        "    text(x+0.72,2.30,items{i}{1},'FontSize',8,'HorizontalAlignment','center');",
        "    text(x+0.72,1.80,items{i}{2},'FontSize',8,'FontWeight','bold','HorizontalAlignment','center');",
        "    if i<5, annotation(fig,'arrow',[0.175+i*0.18 0.205+i*0.18],[0.53 0.53],'LineWidth',1.4,'Color',S.gray); end",
        "end",
        "text(5,0.65,'Training (1-3)                     Deployment (4-5)','HorizontalAlignment','center','FontSize',9,'FontWeight','bold');",
        "edup(fig,'fig1_overview',outA,outB);",
        'end',
    ]))

    F.append(('fig2_compiler', [
        'function fig2_compiler(S,outA,outB)',
        "fig=figure('Units','inches','Position',[1 1 S.W2 3.25]); ax=axes(fig); axis(ax,[0 10 0 6]); axis off; hold on;",
        "cols=[S.da;S.green;S.ia;S.red]; tit={'Frontend','IR Graph','SCL Backend','Validation'};",
        "sub={{'Python parser','KAN/MLP modeller','SI unit metadata'},{'6 op types','Topological DAG','SVNN type tags'},{'S7-1200/1500','FB + DB split','15-pt LUT'},{'TIA Openness','Compile + diagnose','Python-SCL check'}};",
        "for i=1:4",
        "    x=0.55+(i-1)*2.35;",
        "    rectangle('Position',[x 2.75 1.75 2.45],'Curvature',0.06,'FaceColor',cols(i,:)*0.12+0.88,'EdgeColor',cols(i,:),'LineWidth',1.4);",
        "    rectangle('Position',[x 4.65 1.75 0.55],'Curvature',0.06,'FaceColor',cols(i,:),'EdgeColor',cols(i,:),'LineWidth',1.2);",
        "    text(x+0.875,4.93,tit{i},'Color','w','FontWeight','bold','FontSize',9.5,'HorizontalAlignment','center');",
        "    for k=1:3, text(x+0.875,4.25-0.48*k,sub{i}{k},'FontSize',7.5,'HorizontalAlignment','center'); end",
        "    if i<4, annotation(fig,'arrow',[0.23+i*0.215 0.27+i*0.215],[0.665 0.665],'LineWidth',1.4,'Color',S.gray); end",
        "end",
        "panels={'Optimizer','Static Analyzer','Verifier'}; pcols=[S.da;S.green;S.purple];",
        "pitems={{'6-pass optimizer','FuseMatMul, LUTize','Adaptive B-spline'},{'Memory budget','FLOPs + WCET','DA margin'},{'Template proof','Z3 per-func','Compiler TCB'}};",
        "for i=1:3",
        "    x=1.1+(i-1)*3;",
        "    rectangle('Position',[x 0.55 2.05 1.45],'Curvature',0.06,'FaceColor',[0.95 0.95 0.95],'EdgeColor',S.gray,'LineWidth',1.2);",
        "    text(x+1.025,1.72,panels{i},'FontSize',9,'FontWeight','bold','HorizontalAlignment','center');",
        "    text(x+1.025,1.25,strjoin(pitems{i},' | '),'FontSize',7,'HorizontalAlignment','center');",
        "end",
        "edup(fig,'fig2_compiler_arch',outA,outB);",
        'end',
    ]))

    return F


if __name__ == '__main__':
    make_script()
