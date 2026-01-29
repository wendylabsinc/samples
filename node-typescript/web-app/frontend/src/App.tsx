import { useState } from "react";
import { ShaderAnimation } from "@/components/ui/shader-animation";
import { Button } from "@/components/ui/button";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

interface Car {
  name: string;
  make: string;
  year: number;
  color: string;
  createdAt: string;
}

function App() {
  const [cars, setCars] = useState<Car[]>([]);
  const [loading, setLoading] = useState(false);

  const fetchCar = async () => {
    setLoading(true);
    try {
      const response = await fetch("/api/random-car");
      const car: Car = await response.json();
      setCars((prev) => [car, ...prev].slice(0, 10));
    } catch (error) {
      console.error("Failed to fetch car:", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="relative flex min-h-screen w-full flex-col items-center overflow-hidden bg-blue-700">
      <div className="relative flex h-[400px] w-full flex-col items-center justify-center">
        <ShaderAnimation />
        <span className="absolute pointer-events-none z-10 text-center text-7xl leading-none font-semibold tracking-tighter whitespace-pre-wrap text-white">
          WendyOS
        </span>
      </div>

      <div className="z-10 flex flex-col items-center gap-6 p-8">
        <Button
          onClick={fetchCar}
          disabled={loading}
          size="lg"
          variant="secondary"
          className="text-lg"
        >
          {loading ? "Fetching..." : "Fetch Car"}
        </Button>

        {cars.length > 0 && (
          <div className="w-full max-w-4xl rounded-lg bg-white/10 backdrop-blur-sm p-4">
            <Table>
              <TableHeader>
                <TableRow className="border-white/20 hover:bg-white/5">
                  <TableHead className="text-white">Name</TableHead>
                  <TableHead className="text-white">Make</TableHead>
                  <TableHead className="text-white">Year</TableHead>
                  <TableHead className="text-white">Color</TableHead>
                  <TableHead className="text-white">Created At</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {cars.map((car, index) => (
                  <TableRow
                    key={`${car.createdAt}-${index}`}
                    className="border-white/20 hover:bg-white/5"
                  >
                    <TableCell className="text-white font-medium">
                      {car.name}
                    </TableCell>
                    <TableCell className="text-white">{car.make}</TableCell>
                    <TableCell className="text-white">{car.year}</TableCell>
                    <TableCell className="text-white">
                      <div className="flex items-center gap-2">
                        <div
                          className="w-4 h-4 rounded border border-white/30"
                          style={{ backgroundColor: car.color }}
                        />
                        {car.color}
                      </div>
                    </TableCell>
                    <TableCell className="text-white/70 text-sm">
                      {new Date(car.createdAt).toLocaleTimeString()}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
