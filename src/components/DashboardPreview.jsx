import {
    BarChart,
    Bar,
    PieChart, 
    Pie, 
    Cell,    
    XAxis,
    YAxis,
    Tooltip,
    ResponsiveContainer,
    CartesianGrid,
    LineChart,
    Line
} from "recharts";

const demoData = [
    { name: "Jan", value: 400 },
    { name: "Feb", value: 300 },
    { name: "Mar", value: 500 },
    { name: "Apr", value: 280 },
    { name: "May", value: 600 },
    { name: "Jun", value: 450 },
    { name: "Jul", value: 700 },
    { name: "Aug", value: 550 }
];

const DashboardPreview = () => {
    return (
        <div className="relative min-h-screen bg-black text-white overflow-hidden">

            {/* Background dots */}
            <div className="absolute inset-0 bg-[radial-gradient(circle_at_1px_1px,#1f2937_1px,transparent_0)] bg-[size:24px_24px] opacity-40" />

            {/* Floating blur */}
            <div className="absolute -top-40 -left-40 w-[500px] h-[500px] bg-indigo-500/10 rounded-full blur-3xl" />
            <div className="absolute top-1/3 right-0 w-[400px] h-[400px] bg-cyan-500/10 rounded-full blur-3xl" />

            {/* Dashboard Grid */}
            <div className="relative p-10 grid grid-cols-2 gap-8">

                {/* Card 1 - Pie Chart */}
                <div className="bg-white/5 backdrop-blur-xl border border-white/10 rounded-2xl p-5 shadow-xl">
                    <h3 className="text-sm text-gray-400 mb-4">Sales Distribution</h3>

                    <ResponsiveContainer width="100%" height={160}>
                        <PieChart>
                            <Pie
                                data={demoData.slice(0, 5)}
                                dataKey="value"
                                nameKey="name"
                                cx="50%"
                                cy="50%"
                                innerRadius={40}
                                outerRadius={70}
                                paddingAngle={4}
                            >
                                {demoData.slice(0, 5).map((entry, index) => (
                                    <Cell
                                        key={`cell-${index}`}
                                        fill={["#6366F1", "#22D3EE", "#22C55E", "#F59E0B", "#EF4444"][index % 5]}
                                    />
                                ))}
                            </Pie>

                            <Tooltip />
                        </PieChart>
                    </ResponsiveContainer>
                </div>

                {/* Card 2 - Line Chart */}
                <div className="bg-white/5 backdrop-blur-xl border border-white/10 rounded-2xl p-5 shadow-xl">
                    <h3 className="text-sm text-gray-400 mb-4">Growth Trend</h3>

                    <ResponsiveContainer width="100%" height={160}>
                        <LineChart data={demoData.slice(0, 6)}>
                            <XAxis dataKey="name" hide />
                            <YAxis hide />
                            <Tooltip />
                            <Line
                                type="monotone"
                                dataKey="value"
                                stroke="#22D3EE"
                                strokeWidth={3}
                                dot={{ r: 4 }}
                            />
                        </LineChart>
                    </ResponsiveContainer>
                </div>

                {/* Card 3 - Large Overview */}
                <div className="col-span-2 bg-white/5 backdrop-blur-xl border border-white/10 rounded-2xl p-6 shadow-xl">
                    <h3 className="text-sm text-gray-400 mb-5">Revenue Overview</h3>

                    <ResponsiveContainer width="100%" height={260}>
                        <BarChart data={demoData}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                            <XAxis dataKey="name" />
                            <YAxis />
                            <Tooltip />
                            <Bar dataKey="value" fill="#22C55E" radius={[10, 10, 0, 0]} />
                        </BarChart>
                    </ResponsiveContainer>
                </div>

            </div>
        </div>
    );
};

export default DashboardPreview;
