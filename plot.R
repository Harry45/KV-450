library(corrplot)

nselect = 10
fontSize = 15
COL2(diverging = c("RdBu", "BrBG", "PiYG", "PRGn", "PuOr", "RdYlBu"), n = 200)

nsamples = 15000
height = 500
width = 500
cbarWidth = 0.10

for (set in list(2, 3, 4, 5, 6))
{
	fname_1 = paste('Correlations/set_', toString(nsamples), '_', toString(set), '/z0.1t0.3.csv', sep='')
	fname_2 = paste('Correlations/set_', toString(nsamples), '_', toString(set), '/z0.3t0.5.csv', sep='')
	fname_3 = paste('Correlations/set_', toString(nsamples), '_', toString(set), '/z0.5t0.7.csv', sep='')
	fname_4 = paste('Correlations/set_', toString(nsamples), '_', toString(set), '/z0.7t0.9.csv', sep='')
	fname_5 = paste('Correlations/set_', toString(nsamples), '_', toString(set), '/z0.9t1.2.csv', sep='')

	sample_1 = read.csv(fname_1)
	sample_2 = read.csv(fname_2)
	sample_3 = read.csv(fname_3)
	sample_4 = read.csv(fname_4)
	sample_5 = read.csv(fname_5)

	heights_1 = sample_1[3:length(sample_1)]
	heights_2 = sample_2[3:length(sample_2)]
	heights_3 = sample_3[3:length(sample_3)]
	heights_4 = sample_4[3:length(sample_4)]
	heights_5 = sample_5[3:length(sample_5)]

	matrix_1 = cor(t(heights_1))
	matrix_2 = cor(t(heights_2))
	matrix_3 = cor(t(heights_3))
	matrix_4 = cor(t(heights_4))
	matrix_5 = cor(t(heights_5))


	path = paste('plots/set_', toString(nsamples), '_', toString(set), "/1_correlation.pdf", sep ='')
	png(height=height, width=width, pointsize=fontSize, file=path)
	corrplot(matrix_1[0:nselect, 0:nselect], method = "ellipse", type = "lower", diag = FALSE, col = COL2('PRGn'), cl.ratio=cbarWidth, tl.pos='n')

	path = paste('plots/set_', toString(nsamples), '_', toString(set), "/2_correlation.pdf", sep ='')
	png(height=height, width=width, pointsize=fontSize, file=path)
	corrplot(matrix_2[0:nselect, 0:nselect], method = "ellipse", type = "lower", diag = FALSE, col = COL2('PRGn'), cl.ratio=cbarWidth, tl.pos='n')

	path = paste('plots/set_', toString(nsamples), '_', toString(set), "/3_correlation.pdf", sep ='')
	png(height=height, width=width, pointsize=fontSize, file=path)
	corrplot(matrix_3[0:nselect, 0:nselect], method = "ellipse", type = "lower", diag = FALSE, col = COL2('PRGn'), cl.ratio=cbarWidth, tl.pos='n')

	path = paste('plots/set_', toString(nsamples), '_', toString(set), "/4_correlation.pdf", sep ='')
	png(height=height, width=width, pointsize=fontSize, file=path)
	corrplot(matrix_4[0:nselect, 0:nselect], method = "ellipse", type = "lower", diag = FALSE, col = COL2('PRGn'), cl.ratio=cbarWidth, tl.pos='n')

	path = paste('plots/set_', toString(nsamples), '_', toString(set), "/5_correlation.pdf", sep ='')
	png(height=height, width=width, pointsize=fontSize, file=path)
	corrplot(matrix_5[0:nselect, 0:nselect], method = "ellipse", type = "lower", diag = FALSE, col = COL2('PRGn'), cl.ratio=cbarWidth, tl.pos='n')
}