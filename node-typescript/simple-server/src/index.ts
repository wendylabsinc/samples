import express, { Request, Response } from "express";

const app = express();
const port = 3000;

app.use(express.json());

app.get("/", (_req: Request, res: Response) => {
  res.send("Hello, World!");
});

app.get("/hello/:name", (req: Request, res: Response) => {
  res.send(`Hello, ${req.params.name}!`);
});

interface CreateUserBody {
  username: string;
}

interface User {
  id: number;
  username: string;
}

app.post("/users", (req: Request<{}, User, CreateUserBody>, res: Response) => {
  const user: User = {
    id: 1,
    username: req.body.username,
  };
  res.status(201).json(user);
});

app.listen(port, () => {
  console.log(`Server running on http://localhost:${port}`);
});
