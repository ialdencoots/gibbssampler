from numpy.random.mtrand import dirichlet
from numpy.random.mtrand import multinomial
from numpy.random.mtrand import randint
from math import log
from math import exp
from math import lgamma
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

alphabet = ['A', 'C', 'G', 'T']
mparam = [1, 2, 2, 1]
backparam = [1] * 4
#mparam = [34, 2, 6, 21]
#backparam = [2, 12, 9, 3]
num_trials = 10
mlength = 10

def generate(num_seq, seq_length, alphabet, m_word_length, m_word_param, background_param):
	magic_thetas = [dirichlet(m_word_param) for j in range(m_word_length)]
	background_theta = dirichlet(background_param)
	sequences = []
	starts = []
	for k in range(num_seq):
		background_onehots = [multinomial(1, background_theta) for x in range(seq_length - m_word_length)]
		background = [alphabet[t] for t in [i.tolist().index(1) for i in background_onehots]]
		#background = [alphabet[t].lower() for t in [i.tolist().index(1) for i in background_onehots]]
		magic_onehots = [multinomial(1, theta) for theta in magic_thetas]
		magic_word = [alphabet[j] for j in [i.tolist().index(1) for i in magic_onehots]]
		start_pos = randint(seq_length - m_word_length)
		background[start_pos : start_pos] = magic_word
		sequences.append(background)
		starts.append(start_pos)
	#print starts
	ans = []
	ans.append(starts)
	ans.append(sequences)
	return ans

def gen_seq(num_seq, seq_length, m_word_length):
	return generate(num_seq, seq_length, alphabet, m_word_length, mparam, backparam)

def gen():
	return gen_seq(1000,100,mlength)

def gibbssample(num_iters, pos_sequences, alphabet, m_word_length, m_word_param, background_param):
	#print sequences
	sequences = pos_sequences[1]
	M = len(alphabet) # Alphabet size
	K = len(sequences) # Num sequences
	N = len(sequences[0]) # Seq length
	alph_map = {alphabet[m] : m for m in range(M)}
	iter_logpost = [[] for x in range(2)]

	# Initialize hidden word starting locations
	R = randint(0, N - m_word_length, K).tolist()

	# Calculate sum of alphas for magic word and background distributions
	A = float(sum(m_word_param))
	A_back = float(sum(background_param))

	# Get magic word and background symbol counts
	N_m = [[0.0] * m_word_length for x in range(M)]
	bg_m = [0.0] * M
	for i in range(K):
		for x in range(N):
			if x >= R[i] and x < R[i] + m_word_length:
				N_m[alph_map[sequences[i][x].upper()]][x-R[i]] += 1
			else:
				bg_m[alph_map[sequences[i][x].upper()]] += 1

	# Begin iterations
	for l in range(num_iters):
		to_exclude = range(K)
		for s in range(K):
			# Select sequence to exclude
			exclude_indx = randint(len(to_exclude))
			z = to_exclude[exclude_indx]
			del to_exclude[exclude_indx]

			# Update counts for excluding excluded sequence
			for x in range(N):
				if x >= R[z] and x < R[z] + m_word_length:
					N_m[alph_map[sequences[z][x].upper()]][x-R[z]] -= 1
				else:
					bg_m[alph_map[sequences[z][x].upper()]] -= 1

			P = [[0.0] * m_word_length for x in range(M)]
			P_bg = [0.0] * M

			# Calculate log conditional P(s_(z,j) = m | s_(j,-z), alpha) for each symbol and m_word pos
			for m in range(M):
				P_bg[m] = log((bg_m[m] + background_param[m]) / (A_back + K*(N-m_word_length) - 1))
				for j in range(m_word_length):
					P[m][j] = log((N_m[m][j] + m_word_param[m]) / (A + K - 1))

			# Calculate posterior over each starting position in s_z
			r_bg_z = sum([P_bg[alph_map[m.upper()]] for m in sequences[z]])
			r = [r_bg_z] * (N - m_word_length)

			for s_pos in range(N - m_word_length):
				for j in range(m_word_length):
					idx = s_pos + j
					r[s_pos] -= P_bg[alph_map[sequences[z][idx].upper()]]
					r[s_pos] += P[alph_map[sequences[z][idx].upper()]][j]

			# Normalize conditionals
			probs = [exp(x) for x in r]
			normalizer = sum(probs)
			probs = [x/normalizer for x in probs]

			# Update starting position for s_z
			R[z] = multinomial(1,probs).tolist().index(1)

			# Update counts for updating starting position of excluded sequence
			for x in range(N):
				if x >= R[z] and x < R[z] + m_word_length:
					N_m[alph_map[sequences[z][x].upper()]][x-R[z]] += 1
				else:
					bg_m[alph_map[sequences[z][x].upper()]] += 1

		# Calculate posterior
		log_post = lgamma(A_back) - lgamma(K*(N - m_word_length) + A_back)
		for m in range(M):
			log_post += lgamma(bg_m[m] + background_param[m]) - lgamma(background_param[m])
		for j in range(m_word_length):
			log_post += lgamma(A) - lgamma(K + A)
			for m in range(M):
				log_post += lgamma(N_m[m][j] + m_word_param[m]) - lgamma(m_word_param[m])

		iter_logpost[0].append(l)
		iter_logpost[1].append(log_post)

	#print[abs(pos_sequences[0][i] - R[i]) < 1 for i in range(len(R))]
	#print R

	return [R,iter_logpost]

#Plots the results of multiple initializations of one generated dataset.
def samp(save_name, iters, wordlength, num_seq, seq_len):
	to_plot = []
	pos_seqs = gen_seq(num_seq, seq_len, wordlength)
	K = len(pos_seqs[1])
	N = len(pos_seqs[1][0])
	for i in range(num_trials):	
		to_plot.append(gibbssample(iters,pos_seqs,alphabet,wordlength,mparam,backparam)[1])
	for i in range(num_trials):
		plt.plot(to_plot[i][0], to_plot[i][1])
	plt.xlabel('Iterations')
	plt.ylabel('log P(D|R,alpha)')
	plt.title(str(num_trials) + ' initializations: '+ str(K) + ' sequences of length ' + str(N) + ', magic word length: ' + str(wordlength))
	plt.xscale('log')
	plt.subplots_adjust(left = .13)
	plt.savefig('alph2' + save_name)
	plt.close()
	#plt.show()
	return

#Everything below this is for attempted comparisons of input parameters.

def comparewordlengths(save_name, iters, wordlengths, num_seq, seq_len):
	to_plot = []
	for i in range(len(wordlengths)):	
		pos_seqs = gen_seq(num_seq, seq_len, wordlengths[i])
		K = len(pos_seqs[1])
		N = len(pos_seqs[1][0])
		to_plot.append(gibbssample(iters,pos_seqs,alphabet,wordlengths[i],mparam,backparam))
	for i in range(len(wordlengths)):
		plt.plot(to_plot[i][0], to_plot[i][1], label=str(wordlengths[i]))
	plt.xlabel('Iterations')
	plt.ylabel('log P(D|R,alpha)')
	plt.title(str(K) + ' sequences of length ' + str(N) + ', varying magic word lengths')
	plt.xscale('log')
	plt.subplots_adjust(left = .13)
	plt.legend(loc=2, title='Hidden word length')
	plt.savefig(save_name)
	plt.close()
	#plt.show()
	return

def comparenumseqs(save_name, iters, wordlength, num_seqs, seq_len):
	to_plot = []
	for i in range(len(num_seqs)):	
		pos_seqs = gen_seq(num_seqs[i], seq_len, wordlength)
		K = len(pos_seqs[1])
		N = len(pos_seqs[1][0])
		to_plot.append(gibbssample(iters,pos_seqs,alphabet,wordlength,mparam,backparam))
	for i in range(len(num_seqs)):
		m = max([-x for x in to_plot[i][1]])
		plt.plot(to_plot[i][0], [x/m for x in to_plot[i][1]], label=str(num_seqs[i]))
	plt.xlabel('Iterations')
	plt.ylabel('log P(D|R,alpha) / max(-log P(D|R,alpha))')
	plt.title('Sequences of length ' + str(N) + ', magic word length: ' + str(wordlength))
	plt.xscale('log')
	plt.subplots_adjust(left = .13)
	plt.legend(loc=4, title='Number of sequences')
	plt.savefig(save_name)
	plt.close()
	#plt.show()
	return

params = [[1,2,2,1],[1,4,4,1],[1,8,8,1],[1,16,16,1]]
def compareparams(save_name, iters, wordlength, num_seq, seq_len):
	to_plot = []
	for i in range(len(params)):	
		pos_seqs = generate(num_seq, seq_len, alphabet, wordlength, params[i], backparam)
		K = len(pos_seqs[1])
		N = len(pos_seqs[1][0])
		to_plot.append(gibbssample(iters,pos_seqs,alphabet,wordlength,params[i],backparam))
	for i in range(len(params)):
		plt.plot(to_plot[i][0], to_plot[i][1], label=str(params[i]))
	plt.xlabel('Iterations')
	plt.ylabel('log P(D|R,alpha)')
	plt.title(str(K) + ' sequences of length ' + str(N) + ', magic word length: '+str(wordlength))
	plt.xscale('log')
	plt.subplots_adjust(left = .13)
	plt.legend(loc=2, title='Hidden word parameters')
	plt.savefig(save_name)
	plt.close()
	#plt.show()
	return
