import pip
import csv
import matplotlib.pyplot as plt

pct = []
sms = []

with open('bench.csv', newline='\n') as csvfile:
    bench_csv = csv.reader(csvfile, delimiter=',')
    next(bench_csv, None)
    for row in bench_csv:
        sms.append(int(row[0]))
        pct.append(float(row[3]))

ratio_to_full_machine = [sm / 80.0 * 100.0 for sm in sms]
# plt.plot(sms, optimal, 'b-', label='optimal')

plt.plot(ratio_to_full_machine, pct, 'go--', label='achieved')
plt.xlabel('% of total SMs')
plt.ylabel('% to reference')
plt.title('Elapsed time for different SMs vs. full GPU (80 SMs)')
plt.legend()
plt.grid( linestyle = '--', linewidth = 0.5)
plt.show()
plt.savefig('bench.png')

