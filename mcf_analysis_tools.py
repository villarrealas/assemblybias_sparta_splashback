import numpy as np
from numpy.lib.recfunctions import append_fields, merge_arrays
import halotools.mock_observables as mo
import halotools.sim_manager as sm
import scipy.stats as stats

class SpartaCatalog(object):
	"""An ingested Sparta catalog with routines for carrying out necessary calculations on that catalog.
	"""


	def __init__(self, inpath, minmass, maxmass, masschoice, lbox):
		"""Read a halo catalog generated in the sparta format using the tools from
			halotools and generate any custom variables we need.
			ARGS:
			inpath (str): path to the input sparta ascii file.
			minmass (float): minimum mass to cut the dataset at in Msun/h
			maxmass (float): maximum mass to cut the dataset at in Msun/h
			masschoice (str): string matching the dictionary value for a chosen mass definition
				to apply cuts on.
			lbox (float): the size of the box in Mpc/h for any calculations.
			RETURN:
			outdata (structued array): A numpy array of the initially modified data.
		"""
		# first define a dictionary of properties we are interested in.
		rs_dict = {'halo_id':(1,'i8'), 'halo_upid':(6,'i8'), 'halo_M200b':(10,'f8'), 'halo_R200b':(11,'f8'),
				 	'halo_rs':(12,'f8'), 'halo_vmax':(16,'f8'), 'halo_x':(17,'f8'), 'halo_y':(18,'f8'),
					'halo_z':(19,'f8'), 'halo_spin':(26,'f8'), 'halo_c_to_a':(47,'f8'), 'halo_Acc_Rate_100Myr':(66,'f8'),
					'halo_Acc_Rate_TDyne':(67,'f8'), 'halo_Acc_Rate_2TDyne':(68,'f8'), 'halo_Rspstatus':(80,'i8'),
					'halo_upid_mean':(81,'i8'), 'halo_Rsp_mean':(82,'f8'), 'halo_Msp_mean':(84,'f8'),
					'halo_upid_percentile75':(91,'i8'),'halo_Rsp_percentile75':(92,'f8'), 'halo_Msp_percentile75':(94,'f8'),
					'halo_upid_percentile87':(96,'i8'),'halo_Rsp_percentile87':(97,'f8'),'halo_Msp_percentile87':(99,'f8')}
		# Now we need to start the reader and create the data. Cut anything without Sparta information.
		reader = sm.TabularAsciiReader(inpath, rs_dict, row_cut_eq_dict={'halo_Rspstatus':0},
									row_cut_min_dict={masschoice:minmass}, row_cut_max_dict={masschoice:maxmass})
		self.data = reader.read_ascii()
		self.lbox = lbox
		self.cores = 1
	def calculate_cV(self, mdefchoice):
		""" Calculate cV given a mass definition choice. Definition should string following M/R in definition.
		"""
		gnewton = 4.302e-6 # units for calculating cV
		cV = self.data['halo_vmax']/np.sqrt(gnewton*self.data['halo_M'+mdefchoice]/self.data['halo_R'+mdefchoice])
		return cV
	def add_standard_properties(self):
		""" Augments the dataset with a series of standard properties that we use in assembly bias
		calculations.
		"""
		# 5 definitions of halo concentration.
		cNFW200b = self.data['halo_R200b']/self.data['halo_rs']
		cV200b = self.calculate_cV('200b')
		cVsp_mean = self.calculate_cV('sp_mean')
		cVsp_percentile75 = self.calculate_cV('sp_percentile75')
		cVsp_percentile87 = self.calculate_cV('sp_percentile87')

		# Plus several halo size ratios.
		sizeratiosp87_200b = self.data['halo_Rsp_percentile87']/self.data['halo_R200b']
		sizeratiosp75_200b = self.data['halo_Rsp_percentile75']/self.data['halo_R200b']
		sizeratiospmean_200b = self.data['halo_Rsp_mean']/self.data['halo_R200b']
		sizeratiosp87_spmean = self.data['halo_Rsp_percentile87']/self.data['halo_Rsp_mean']

		# And the same mass ratios
		massratiosp87_200b = self.data['halo_Msp_percentile87']/self.data['halo_M200b']
		massratiosp75_200b = self.data['halo_Msp_percentile75']/self.data['halo_M200b']
		massratiospmean_200b = self.data['halo_Msp_mean']/self.data['halo_M200b']
		massratiosp87_spmean = self.data['halo_Msp_percentile87']/self.data['halo_Msp_mean']

		# And randoms for calculation ease
		uniformrands = np.random.uniform(0,1,len(self.data))

		# Add these all into the data.
		self.data = append_fields(self.data, ('halo_cNFW200b', 'halo_cV200b', 'halo_cVsp_mean',
				'halo_cVsp_percentile75', 'halo_cVsp_percentile87', 'halo_sizeratiosp87_200b',
				'halo_sizeratiosp75_200b', 'halo_sizeratiospmean_200b', 'halo_sizeratiosp87_spmean',
				'halo_massratiosp87_200b', 'halo_massratiosp75_200b', 'halo_massratiospmean_200b',
				'halo_massratiosp87_spmean', 'err_rands'), (cNFW200b, cV200b, cVsp_mean,
				cVsp_percentile75, cVsp_percentile87, sizeratiosp87_200b, sizeratiosp75_200b,
				sizeratiospmean_200b, sizeratiosp87_spmean, massratiosp87_200b, massratiosp75_200b,
				massratiospmean_200b, massratiosp87_spmean, uniformrands), usemask=False)
	def calculate_mcf(self, markname, norm=True, normbinning=20, mdefchoice='200b', rbinning=[3,20,10],
						pidchoice='upid', recalc=False):
		""" Calculate the marked correlation function given a mark. Stores
			the results existing values. The normalization flag allows for
			whether or not to normalize the input marks with ranks.
            The normbinning argument says how many bins to use in normalization.
            Mdefchoice defines the mass definition in which to bin. The binning
			argument consists of lower limit, upper limit, and number of bins
			for the marked correlation function.
		"""
		if hasattr(self, 'mcf_'+markname) and recalc==False:
			print('Previously calculated this mark! Did you mean to include recalc==True?\n')
			return getattr(self, 'mcf_'+markname)
		else:
			mark = self.data['halo_'+markname][self.data['halo_'+pidchoice]==-1]
			# first optional argument: whether or not to normalize.
			if norm==True:
				logmass = np.log10(self.data['halo_M'+mdefchoice][self.data['halo_'+pidchoice]==-1])
				bins = np.linspace(logmass.min(), logmass.max(), normbinning)
				idx = np.digitize(logmass, bins)
				for i in range(1, normbinning+1):
					ranks = np.zeros(len(logmass[idx==i]))
					ranks = stats.rankdata(mark[idx==i], 'average') / len(ranks)
					mark[idx==i]=ranks
			# set up binning information
			minlog = np.log10(rbinning[0])
			maxlog = np.log10(rbinning[1])
			nstep = rbinning[2]
			steplog = (maxlog - minlog) / (nstep-1)
			logbins = np.linspace(minlog, maxlog, num=nstep+1)
			binmids = np.zeros(nstep)
			for i in range(0,nstep):
				binmids[i] = (logbins[i]+logbins[i+1])/2.
			self.binmids_last = binmids
			pos = np.vstack((self.data['halo_x'][self.data['halo_'+pidchoice]==-1], self.data['halo_y'][self.data['halo_'+pidchoice]==-1], self.data['halo_z'][self.data['halo_'+pidchoice]==-1])).T

			# check if we have calculated errors before:
			if not hasattr(self, 'mcf_lowererr'):
				nrand = 200
				mcfn_rand = np.zeros((nstep, nrand))
				for i in range(0, nrand):
					randerr = np.random.permutation(self.data['err_rands'][self.data['halo_'+pidchoice]==-1])
					mcfn_rand[:,i] = (mo.marked_tpcf(pos, 10**logbins, marks1=randerr,
						period=self.lbox, normalize_by='number_counts', weight_func_id=1,
						num_threads=self.cores)-.5**2)/(1./12.)
				self.mcf_uppererr = [np.percentile(mcfn_rand[i,:],98) for i in range(nstep)]
				self.mcf_lowererr = [np.percentile(mcfn_rand[i,:],2) for i in range(nstep)]

			mcf_temp = mo.marked_tpcf(pos, 10**logbins, marks1=mark, period=self.lbox, normalize_by='number_counts',
										weight_func_id=1, num_threads=self.cores)
			mcfn_temp = (mcf_temp - .5**2)/(1./12.)
			setattr(self,'mcf_'+markname, mcfn_temp) 
			return mcfn_temp
