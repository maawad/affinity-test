import pip
import csv
import matplotlib.pyplot as plt

pct = []
sms = []

with open('bench_amd.csv', newline='\n') as csvfile:
    bench_csv = csv.reader(csvfile, delimiter=',')
    next(bench_csv, None)
    for row in bench_csv:
        sms.append(int(row[0]))
        pct.append(float(row[3]))

ratio_to_full_machine = [sm / 42 * 100.0 for sm in sms]
# plt.plot(sms, optimal, 'b-', label='optimal')

plt.plot(ratio_to_full_machine, pct, 'go--', label='achieved')
plt.xlabel('% of total CUs')
plt.ylabel('ratio to total CUs')
plt.title('Elapsed time for different CUs vs. full GPU (42 CUs)')
plt.legend()
plt.grid( linestyle = '--', linewidth = 0.5)
plt.show()
plt.savefig('bench.png')

